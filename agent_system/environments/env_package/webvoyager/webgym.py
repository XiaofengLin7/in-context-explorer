import platform
import time
import json
import re
import os
import shutil
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from gym.spaces import Box, Discrete, Dict as DictSpace, Text
import gym

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait

from .webvoyager.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY
from openai import OpenAI
from .webvoyager.utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only
from .webvoyager.utils_webarena import webarena_login, WEBARENA_DOMAINS
from .webvoyager.utils_eval import webarena_eval


class WebVoyagerEnv(gym.Env):
    """
    Gym wrapper for WebVoyager/WebArena Environment
    """
    
    def __init__(self,
                 api_key: str, # only needed for processing pdf files
                 api_model: str = "gpt-4-vision-preview",
                 env_config = None,
                 headless: bool = True,
                 text_only: bool = False,
                 window_width: int = 1024,
                 window_height: int = 768,
                 download_dir: str = "downloads",
                 max_attached_imgs: int = 1,
                 fix_box_color: bool = True,
                 save_accessibility_tree: bool = True,
                 force_device_scale: bool = False,
                 batch_id: int = 0,
                 num_containers_per_machine: int = 1,
                 worker_id: Optional[str] = None):
        """
        Initialize the WebVoyager environment.
        
        Args:
            api_key: OpenAI API key
            api_model: OpenAI model to use
            headless: Whether to run browser in headless mode
            text_only: Whether to use text-only mode (accessibility tree)
            window_width: Browser window width
            window_height: Browser window height
            download_dir: Directory for downloads
            max_attached_imgs: Maximum number of images to attach
            fix_box_color: Whether to fix box colors
            save_accessibility_tree: Whether to save accessibility tree
            force_device_scale: Whether to force device scale
            batch_id: Batch ID for multi-container setups
            num_containers_per_machine: Number of containers per machine
        """
        super().__init__()
        
        # Store configuration
        self.api_key = api_key
        self.api_model = api_model # to process pdf files
        
        self.env_config = env_config
        if self.env_config.get("webarena", False):
            self.webarena_host = env_config.webarena_host
        else:
            self.webarena_host = None
        
        self.headless = headless
        self.text_only = text_only
        self.window_width = window_width
        self.window_height = window_height
        self.download_dir = download_dir
        #self.max_attached_imgs = max_attached_imgs
        self.fix_box_color = fix_box_color
        self.save_accessibility_tree = save_accessibility_tree
        self.force_device_scale = force_device_scale
        self.batch_id = batch_id
        self.num_containers_per_machine = num_containers_per_machine
        self.url_mapping = []
        self.worker_id = worker_id
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Initialize browser options
        self.options = self._driver_config()
        
        # Environment state
        self.driver = None
        self.task = None
        self.timestep = 0
        self.download_files = []
        self.fail_obs = ""
        self.pdf_obs = ""
        self.warn_obs = ""
        #self.accumulate_prompt_token = 0
        #self.accumulate_completion_token = 0

        self.img_path = None
        #self.obs_info = None
        self.accessibility_tree = None
        #self.web_eles_text = None
        #self.web_eles = None
        self.messages = []
        #self.init_msg = None

        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)
        
    def _driver_config(self):
        """Configure Chrome driver options."""
        options = webdriver.ChromeOptions()
        
        if self.save_accessibility_tree:
            self.force_device_scale = True
            
        if self.force_device_scale:
            options.add_argument("--force-device-scale-factor=1")
        if self.headless:
            options.add_argument("--headless")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
        # Improve stability in headless/HPC environments
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        # Set window size via Chrome options to avoid hanging issues with set_window_size()
        # This is more reliable, especially in headless mode
        options.add_argument(f"--window-size={self.window_width},{self.window_height}")
        options.add_experimental_option(
            "prefs", {
                "download.default_directory": self.download_dir,
                "plugins.always_open_pdf_externally": True
            }
        )
        return options
    
    def reset(self, task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment with a new task.
        
        Args:
            task: Dictionary containing task information with keys:
                - 'id': task ID
                - 'ques': task question
                - 'web': website URL
            
        Returns:
            observation: Current observation
            info: Additional information
        """
        # Close existing driver if any
        if self.driver is not None:
            self.driver.quit()
            
        # Initialize new driver
        # Window size is set via Chrome options (--window-size) to avoid hanging issues
        # that can occur with driver.set_window_size() in headless mode
        self.driver = webdriver.Chrome(options=self.options)
        
        # Explicit timeouts
        try:
            self.driver.set_page_load_timeout(30)
        except Exception:
            pass
        try:
            self.driver.set_script_timeout(30)
        except Exception:
            pass
        
        # Store current task
        self.task = task
        self.process_task_eval_info()
        self.timestep = 0
        
        # Clear pdf files in download directory
        for filename in os.listdir(self.download_dir):
            file_path = os.path.join(self.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        base_task_dir = os.path.join(self.download_dir, 'task{}'.format(task["id"]), 'batch_id{}'.format(self.batch_id))
        os.makedirs(base_task_dir, exist_ok=True)

        # Create a unique per-worker subdirectory under the task directory
        wid = self.worker_id if self.worker_id is not None else f"pid{os.getpid()}_{int(time.time()*1000)%1000000}"
        self.worker_dir = os.path.join(base_task_dir, f'worker_{wid}')
        os.makedirs(self.worker_dir, exist_ok=True)

        # Preserve base task directory for compatibility, but write files to worker_dir
        self.task_dir = base_task_dir

        self.download_files = []
        self.fail_obs = ""
        self.pdf_obs = ""
        self.warn_obs = ""
        
        # Initialize URL mapping
        self.url_mapping = [(task['web'], task['web'])]
        url = task['web']
        assert url is not None, f"url is None for task {task}"
        # breakpoint()
        # Handle WebArena login if needed (do this before creating messages)
        if self.webarena_host and task.get('web_name') and task['web_name'] in WEBARENA_DOMAINS:
            if "ec2-98-81-119-107.compute-1.amazonaws.com" in url:
                url = url.replace("ec2-98-81-119-107.compute-1.amazonaws.com", self.env_config.common_webarena_host) # replace the starting url with host server
            if "192.222.54.137" in url:
                url = url.replace("192.222.54.137", self.env_config.common_webarena_host) # replace the starting url with host server
            # Replace reddit references in task question (similar to webgym implementation)
            if task['web_name'] == 'reddit':
                task['ques'] = task['ques'].replace("subreddit", "subforum").replace("sub-reddit", "subforum").replace("reddit", "postmill").replace("Reddit", "Postmill")
            
            success, url_mapping, url = webarena_login(
                task['web_name'], 
                url, 
                self.driver, 
                self.webarena_host, 
                batch_id=self.batch_id, 
                num_containers_per_machine=self.num_containers_per_machine
            )

            if not success:
                logging.error(f"[ERROR] WEBARENA LOGIN FAIL for {task['web_name']} and url {url}")
                self.fail_obs = f"WebArena login failed for {task['web_name']}"
                self.starting_url = url
                observation = self.get_observation()
                info = {
                    'task_id': task['id'],
                    'task_question': task['ques'],
                    'task_url': task['web'],
                    'done': True,
                    'iteration': self.timestep
                }
                return observation, info
            else:
                print(f"WebArena login successful for {task['web_name']} and url {url}")
            if len(url_mapping) != 0:
                self.url_mapping = url_mapping
        else:
            print(f"reset failed for {task['web_name']}")
            self.fail_obs = f"not such task."
            self.starting_url = url
            observation = {}
            info = {
                'done': True,
                'iteration': self.timestep
            }
            return observation, info
        
        self.starting_url = url
        print(f"url returned from login: {url}")
        
        # Navigate to the task URL
        try:
            self.driver.get(url)
        except Exception as e:
            logging.error(f"Failed to navigate to the task URL: {e}")
            self.fail_obs = f"Failed to navigate to the task URL: {e}"
            observation = self.get_observation()
            info = {
                'done': True,
                'iteration': self.timestep
            }
            return observation, info
        try:
            self.driver.find_element(By.TAG_NAME, 'body').click()
        except Exception as e:
            logging.warning(f"Could not click body element: {e}")
            pass
        
        # Prevent space key from scrolling
        self.driver.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # get env info
        try:
            if not self.text_only:
                rects, self.web_eles, self.web_eles_text = get_web_element_rect(self.driver, fix_color=self.fix_box_color)
            else:
                accessibility_tree_path = os.path.join(self.worker_dir, 'accessibility_tree{}'.format(self.timestep))
                self.accessibility_tree, self.obs_info = get_webarena_accessibility_tree(self.driver, accessibility_tree_path)

        except Exception as e:
            if not self.text_only:
                logging.error('Driver error when adding set-of-mark.')
                self.fail_obs = "Driver error when adding set-of-mark."
            else:
                logging.error('Driver error when obtaining accessibility tree.')
                self.fail_obs = "Driver error when obtaining accessibility tree."
            logging.error(e)
            
        self.img_path = os.path.join(self.worker_dir, 'screenshot{}.png'.format(self.timestep))
        try:
            self.driver.save_screenshot(self.img_path)
        except Exception as e:
            logging.error(f"Screenshot failed: {e}")
            # Record failure but continue without crashing the worker
            self.fail_obs = self.fail_obs or "Screenshot failed due to unresponsive browser."

        # accessibility tree
        if (not self.text_only) and self.save_accessibility_tree:
            accessibility_tree_path = os.path.join(self.worker_dir, 'accessibility_tree{}'.format(self.timestep))
            self.accessibility_tree, self.obs_info = get_webarena_accessibility_tree(self.driver, accessibility_tree_path)
        
        # Get initial observation
        observation = self.get_observation()
        info = {
            'task_id': task['id'],
            'task_question': task['ques'],
            'task_url': task['web'],
            'done': False,
            'iteration': self.timestep
        }
        
        return observation, info
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Dictionary containing action information:
                - 'action_key': str - one of 'click', 'type', 'scroll', 'wait', 'goback', 'google', 'answer'
                - 'element': int
                - 'content': str
                - 'answer_content': str 
            
        Returns:
            observation: New observation after action
            reward: Reward for the action
            done: Whether the episode is done (terminated or truncated)
            info: Additional information
        """
        self.timestep += 1
        
        # get action info, may have a different structure
        action_key = action["action_key"]

        reward = 0.0
        done = False
        
        self.fail_obs = ""
        self.pdf_obs = ""
        self.warn_obs = ""
        
        # execute action
        try:
            window_handle_task = self.driver.current_window_handle
            self.driver.switch_to.window(window_handle_task)

            if action_key == 'click':
                try:
                    if not self.text_only:
                        click_ele_number = action['element']
                        if click_ele_number >= len(self.web_eles):
                            raise IndexError(f"Element index {click_ele_number} out of range. Available elements: {len(self.web_eles)}")
                        web_ele = self.web_eles[click_ele_number]
                    else:
                        click_ele_number = action['element']
                        if click_ele_number >= len(self.obs_info):
                            raise IndexError(f"Element index {click_ele_number} out of range. Available elements: {len(self.obs_info)}")
                        element_box = self.obs_info[click_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                                element_box[1] + element_box[3] // 2)
                        web_ele = self.driver.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")

                    self._exec_action_click(action, web_ele)

                    # deal with PDF file
                    current_files = sorted(os.listdir(self.download_dir))
                    if current_files != self.download_files:
                        # wait for download finish
                        time.sleep(10)
                        current_files = sorted(os.listdir(self.download_dir))

                        current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in self.download_files and pdf_file.endswith('.pdf')]
                        if current_download_file:
                            print('start to solve pdf')
                            pdf_file = current_download_file[0]
                            pdf_file_path = os.path.join(self.download_dir, pdf_file)
                            try:
                                pdf_obs = get_pdf_retrieval_ans_from_claude(pdf_file_path, self.task['ques'], region_name=self.region, aws_key_id=self.aws_key_id, aws_secret_key=self.aws_secret_key)
                            except:
                                pdf_obs = ""
                            shutil.copy(pdf_file_path, self.task_dir)
                            self.pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                            print("pdf solved", pdf_obs)

                        self.download_files = current_files

                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)
                except Exception as e:
                    self.fail_obs = f"Click action failed: {str(e)}"

            elif action_key == 'wait':
                time.sleep(10)

            elif action_key == 'type':
                try:
                    if not self.text_only:
                        type_ele_number = action['element']
                        if type_ele_number >= len(self.web_eles):
                            raise IndexError(f"Element index {type_ele_number} out of range. Available elements: {len(self.web_eles)}")
                        web_ele = self.web_eles[type_ele_number]
                    else:
                        type_ele_number = action['element']
                        if type_ele_number >= len(self.obs_info):
                            raise IndexError(f"Element index {type_ele_number} out of range. Available elements: {len(self.obs_info)}")
                        element_box = self.obs_info[type_ele_number]['union_bound']
                        print(f"element box: {element_box}")
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                                element_box[1] + element_box[3] // 2)
                        web_ele = self.driver.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    self.warn_obs = self._exec_action_type(action, web_ele)
                    if 'wolfram' in self.task['web']:
                        time.sleep(5)
                except Exception as e:
                    self.fail_obs = f"Type action failed: {str(e)}"

            elif action_key == 'scroll':
                try:
                    if not self.text_only:
                        self._exec_action_scroll(action)
                    else:
                        self._exec_action_scroll(action)
                except Exception as e:
                    self.fail_obs = f"Scroll action failed: {str(e)}"

            elif action_key == 'goback':
                self.driver.back()
                time.sleep(10)

            elif action_key == 'google':
                self.driver.get('https://www.google.com/')
                time.sleep(10)

            elif action_key == 'answer':
                logging.info(action['answer_content'])
                logging.info('finish!!')
                done = True
                reward = self._webarena_eval(action['answer_content'])
            else:
                raise NotImplementedError

        except Exception as e:
            logging.error('driver error info:')
            logging.error(e)
            if 'element click intercepted' not in str(e) and 'IndexError' not in str(type(e).__name__):
                self.fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
            elif 'IndexError' in str(type(e).__name__):
                self.fail_obs = f"Element index out of range: {str(e)}"
            else:
                self.fail_obs = ""
            time.sleep(2)
        
        # in the multi-turn loop in verl-agent, check if max iterations reached
        
        # if self.fail_obs:
        #     done = True
        
        # get env info
        try:
            if not self.text_only:
                rects, self.web_eles, self.web_eles_text = get_web_element_rect(self.driver, fix_color=self.fix_box_color)
            else:
                accessibility_tree_path = os.path.join(self.worker_dir, 'accessibility_tree{}'.format(self.timestep))
                self.accessibility_tree, self.obs_info = get_webarena_accessibility_tree(self.driver, accessibility_tree_path)

        except Exception as e:
            if not self.text_only:
                logging.error('Driver error when adding set-of-mark.')
            else:
                logging.error('Driver error when obtaining accessibility tree.')
            logging.error(e)
            
        self.img_path = os.path.join(self.worker_dir, 'screenshot{}.png'.format(self.timestep))
        self.driver.save_screenshot(self.img_path)

        # accessibility tree
        if (not self.text_only) and self.save_accessibility_tree:
            accessibility_tree_path = os.path.join(self.worker_dir, 'accessibility_tree{}'.format(self.timestep))
            self.accessibility_tree, self.obs_info = get_webarena_accessibility_tree(self.driver, accessibility_tree_path)
        
        # Get new observation
        observation = self.get_observation()
        
        info = {
            'iteration': self.timestep,
            'action_key': action_key,
            'reward': reward,
            'done': done,
            'won': reward == 1.0
        }
        
        return observation, reward, done, info
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from the environment."""
        # Apply URL mapping if webarena is used
        url = self.driver.current_url
        # breakpoint()
        # TODO: figure out why the url is being changed and what happens in the actual observation.
        if self.url_mapping:
            for map_pattern in self.url_mapping:
                url = url.replace(map_pattern[0], map_pattern[1]) if map_pattern[0] in url else url
        
        if self.url_mapping:
            for map_pattern in self.url_mapping:
                self.starting_url = self.starting_url.replace(map_pattern[0], map_pattern[1]) if map_pattern[0] in self.starting_url else self.starting_url
        
        # we can append those we need
        observation = {
                'image': self.img_path,
                'ac_tree': self.accessibility_tree if (not self.text_only and self.save_accessibility_tree) else None,
                'pdf_obs': self.pdf_obs,
                'warn_obs': self.warn_obs,
                'fail_obs': self.fail_obs,
                'task_ques': self.task['ques'] if self.task else "",
                'url': url,
                'web_name': self.task['web_name'] if self.task else "",
                'task_dir': self.task_dir,
                'starting_url': self.starting_url
        }
        return observation
    
    def _webarena_eval(self, answer):
        eval_config = self.task['eval']
        eval_config["webarena_starting_url"] = self.starting_url
        # task content, answer, eval_config, driver, verbose
        return webarena_eval(self.task['ques'], answer, eval_config, self.driver, verbose=True)
    
    def _exec_action_click(self, action, web_ele):
        self.driver.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
        web_ele.click()
        time.sleep(3)

    # TODO: sometimes cannot clear correctly and collapse
    def _exec_action_type(self, action, web_ele):
        self.warn_obs = ""
        type_content = action['content']

        ele_tag_name = web_ele.tag_name.lower()
        ele_type = web_ele.get_attribute("type")
        # outer_html = web_ele.get_attribute("outerHTML")
        if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
            self.warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
        try:
            # Not always work to delete
            try:
                web_ele.clear()
            except Exception as e:
                logging.warning(f"Could not clear input field, try another way to delete")
                # Another way to delete
                if platform.system() == 'Darwin':
                    web_ele.send_keys(Keys.COMMAND + "a")
                else:
                    web_ele.send_keys(Keys.CONTROL + "a")
                web_ele.send_keys(" ")
                web_ele.send_keys(Keys.BACKSPACE)
        except Exception as e:
            logging.warning(f"Could not clear input field: {e}")
            pass

        actions = ActionChains(self.driver)
        actions.click(web_ele).perform()
        actions.pause(1)

        try:
            self.driver.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
        except Exception as e:
            logging.warning(f"Could not set keydown handler: {e}")
            pass

        actions.send_keys(type_content)
        actions.pause(2)

        actions.send_keys(Keys.ENTER)
        actions.perform()
        time.sleep(10)
        return self.warn_obs

    def _exec_action_scroll(self, action):
        scroll_ele_number = action['element']
        scroll_content = action['content']
        if scroll_ele_number == -1:
            if scroll_content == 'down':
                self.driver.execute_script(f"window.scrollBy(0, {self.window_height*2//3});")
            else:
                self.driver.execute_script(f"window.scrollBy(0, {-self.window_height*2//3});")
        else:
            if not self.text_only:
                scroll_ele_number = int(scroll_ele_number)
                if scroll_ele_number >= len(self.web_eles):
                    raise IndexError(f"Element index {scroll_ele_number} out of range. Available elements: {len(self.web_eles)}")
                web_ele = self.web_eles[scroll_ele_number]
            else:
                if scroll_ele_number >= len(self.obs_info):
                    raise IndexError(f"Element index {scroll_ele_number} out of range. Available elements: {len(self.obs_info)}")
                element_box = self.obs_info[scroll_ele_number]['union_bound']
                element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
                web_ele = self.driver.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
            actions = ActionChains(self.driver)
            self.driver.execute_script("arguments[0].focus();", web_ele)
            if scroll_content == 'down':
                actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
            else:
                actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
        time.sleep(3)

    def process_task_eval_info(self):
        """
        Process task evaluation information by replacing old EC addresses 
        with the current webarena host in eval config URLs.
        
        This ensures that webarena_eval can successfully evaluate trajectories
        by using the correct host addresses in reference_url and program_html URLs.
        """
        if self.task.get('eval') is not None and self.webarena_host:
            eval_config = self.task['eval']
            
            # Get common host from env_config if available
            if self.env_config and hasattr(self.env_config, 'common_webarena_host'):
                common_host = self.env_config.common_webarena_host
            else:
                # Fallback: use webarena_host if common_webarena_host is not available
                common_host = self.webarena_host
            
            # Replace old EC addresses in reference_url
            if 'reference_url' in eval_config and eval_config['reference_url']:
                ref_url = eval_config['reference_url']
                if "ec2-98-81-119-107.compute-1.amazonaws.com" in ref_url:
                    eval_config['reference_url'] = ref_url.replace(
                        "ec2-98-81-119-107.compute-1.amazonaws.com", 
                        common_host
                    )
                if "192.222.54.137" in eval_config['reference_url']:
                    eval_config['reference_url'] = eval_config['reference_url'].replace(
                        "192.222.54.137", 
                        common_host
                    )
            
            # Replace old EC addresses in program_html URLs
            if 'program_html' in eval_config and isinstance(eval_config['program_html'], list):
                for program_item in eval_config['program_html']:
                    if 'url' in program_item and program_item['url']:
                        url = program_item['url']
                        if "ec2-98-81-119-107.compute-1.amazonaws.com" in url:
                            program_item['url'] = url.replace(
                                "ec2-98-81-119-107.compute-1.amazonaws.com", 
                                common_host
                            )
                        if "192.222.54.137" in program_item['url']:
                            program_item['url'] = program_item['url'].replace(
                                "192.222.54.137", 
                                common_host
                            )

    def render(self, mode='human'):
        """Render the environment."""
        if self.driver is not None:
            # In headless mode, we can't really render
            if not self.headless:
                pass  # Browser window is already visible
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception as e:
                logging.warning(f"Error closing driver: {e}")
            finally:
                self.driver = None
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()