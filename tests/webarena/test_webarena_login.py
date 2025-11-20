"""Test suite for webarena_login function in utils_webarena.py."""

import sys
import os
from pathlib import Path
from selenium import webdriver
# Add workspace root to Python path for imports
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
import types
import threading
import ray
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

from agent_system.environments.env_package.webvoyager.webgym import WebVoyagerEnv
from agent_system.environments.env_package.webvoyager.webvoyager.utils_webarena import (
    webarena_login,
    WEBARENA_DOMAINS
)

@pytest.fixture
def mock_webvoyager_env():
    with patch('agent_system.environments.env_package.webvoyager.webgym.WebVoyagerEnv') as mock_env:
        mock_env.return_value = MagicMock(spec=WebVoyagerEnv)
        mock_env.return_value.options = _driver_config()
        mock_env.return_value.driver = webdriver.Chrome(options=mock_env.return_value.options)
        yield mock_env

class WebArenaHost(types.SimpleNamespace):
    def __init__(self, common_webarena_host="ec2-3-136-229-169.us-east-2.compute.amazonaws.com"):
        self.reddit = common_webarena_host
        self.shopping_admin = common_webarena_host
        self.gitlab = common_webarena_host
        self.map = common_webarena_host
        self.shopping = common_webarena_host


def _driver_config():
    """Configure Chrome driver options."""
    options = webdriver.ChromeOptions()

    options.add_argument("--force-device-scale-factor=1")

    options.add_argument("--headless")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    # Improve stability in headless/HPC environments
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # Set window size via Chrome options to avoid hanging issues with set_window_size()
    options.add_argument(f"--window-size=1024,768")
    return options


# Test task data
git_task = {
    "sites": ["gitlab"],
    "task_id": 106,
    "require_login": True,
    "storage_state": "./.auth/gitlab_state.json",
    "start_url": "http://ec2-98-81-119-107.compute-1.amazonaws.com:8023",
    "geolocation": None,
    "intent_template": "Display the list of issues in the {{repo}} repository that have labels related to {{label}}",
    "instantiation_dict": {"label": "BUG", "repo": "umano/AndroidSlidingUpPanel"},
    "intent": "Display the list of issues in the umano/AndroidSlidingUpPanel repository that have labels related to BUG",
    "require_reset": False,
    "eval": {
        "eval_types": ["url_match"],
        "reference_answers": None,
        "reference_url": "http://ec2-98-81-119-107.compute-1.amazonaws.com:8023/umano/AndroidSlidingUpPanel/-/issues/?label_name%5B%5D=BUG",
        "program_html": [],
        "url_note": "GOLD in PRED"
    },
    "intent_template_id": 349,
    "web_name": "gitlab",
    "id": "gitlab--106",
    "ques": "Display the list of issues in the umano/AndroidSlidingUpPanel repository that have labels related to BUG",
    "web": "http://ec2-98-81-119-107.compute-1.amazonaws.com:8023"
}

def test_webarena_login(mock_webvoyager_env):
    """Test single GitLab login."""
    web_name = git_task["web_name"]
    url = git_task["web"]
    url = url.replace("ec2-98-81-119-107.compute-1.amazonaws.com", "ec2-3-136-229-169.us-east-2.compute.amazonaws.com")
    driver = mock_webvoyager_env.return_value.driver
    print("driver: ", driver)
    webarena_host = WebArenaHost()
    print("webarena_host: ", webarena_host)
    batch_id = 0
    num_containers_per_machine = 1
    print("num_containers_per_machine: ", num_containers_per_machine)
    success, url_mapping, url = webarena_login(web_name, url, driver, webarena_host, batch_id, num_containers_per_machine)
    assert success


def _login_worker(
    batch_id: int,
    web_name: str,
    url: str,
    webarena_host: WebArenaHost,
    num_containers_per_machine: int,
    results: List[Tuple[int, bool]],
    lock: threading.Lock
) -> None:
    """
    Worker function for concurrent login testing.
    
    Args:
        batch_id: Batch ID for this instance (must be 0, as only batch_id=0 is supported)
        web_name: Web domain name (e.g., 'gitlab')
        url: Starting URL
        webarena_host: WebArena host configuration
        num_containers_per_machine: Number of containers per machine
        results: Shared list to store results
        lock: Thread lock for thread-safe result appending
    """
    try:
        # Create a separate driver instance for this thread
        options = _driver_config()
        driver = webdriver.Chrome(options=options)
        
        try:
            success, url_mapping, final_url = webarena_login(
                web_name=web_name,
                url=url,
                driver_task=driver,
                webarena_host=webarena_host,
                batch_id=batch_id,
                num_containers_per_machine=num_containers_per_machine
            )
            
            with lock:
                results.append((batch_id, success))
            
            print(f"Batch ID {batch_id}: Login {'SUCCESS' if success else 'FAILED'}")
            if success:
                print(f"  URL mapping: {url_mapping}")
                print(f"  Final URL: {final_url}")
        finally:
            # Always close the driver
            driver.quit()
            
    except Exception as e:
        print(f"Batch ID {batch_id}: Exception during login - {e}")
        with lock:
            results.append((batch_id, False))


@pytest.mark.parametrize("num_instances", [2, 3, 4])
def test_concurrent_gitlab_logins(num_instances: int):
    """
    Test multiple instances logging into GitLab concurrently.
    
    This test verifies that:
    1. Multiple instances can login simultaneously without conflicts
    2. All instances use batch_id=0 (only supported value)
    3. All logins succeed independently
    
    Args:
        num_instances: Number of concurrent login instances to test
    """
    web_name = git_task["web_name"]
    base_url = git_task["web"]
    url = base_url.replace(
        "ec2-98-81-119-107.compute-1.amazonaws.com",
        "ec2-3-136-229-169.us-east-2.compute.amazonaws.com"
    )
    
    webarena_host = WebArenaHost()
    num_containers_per_machine = 1  # Only batch_id=0 is supported
    
    # Shared results list with thread lock
    results: List[Tuple[int, bool]] = []
    lock = threading.Lock()
    
    print(f"\n{'='*60}")
    print(f"Testing {num_instances} concurrent GitLab logins (all with batch_id=0)")
    print(f"{'='*60}")
    
    # Execute logins concurrently using ThreadPoolExecutor
    # All instances use batch_id=0 since that's the only supported value
    with ThreadPoolExecutor(max_workers=num_instances) as executor:
        futures = []
        for instance in range(num_instances):
            future = executor.submit(
                _login_worker,
                batch_id=0,  # Only batch_id=0 is supported
                web_name=web_name,
                url=url,
                webarena_host=webarena_host,
                num_containers_per_machine=num_containers_per_machine,
                results=results,
                lock=lock
            )
            futures.append((instance, future))
        
        # Wait for all tasks to complete
        for instance, future in futures:
            try:
                future.result(timeout=120)  # 2 minute timeout per instance
            except Exception as e:
                print(f"Instance {instance}: Future exception - {e}")
                with lock:
                    results.append((instance, False))
    
    # Verify results
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"{'='*60}")
    
    success_count = 0
    for instance, success in sorted(results):
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Instance {instance}: {status}")
        if success:
            success_count += 1
    
    print(f"\nTotal: {success_count}/{num_instances} successful logins")
    
    # Assertions
    assert len(results) == num_instances, f"Expected {num_instances} results, got {len(results)}"
    assert success_count == num_instances, (
        f"Expected all {num_instances} logins to succeed, but only {success_count} succeeded. "
        f"Failed instances: {[instance for instance, s in results if not s]}"
    )
    
    print(f"\n✓ All {num_instances} concurrent logins succeeded!")


@ray.remote
class WebArenaLoginWorker:
    """
    Ray worker class for testing concurrent WebArena logins.
    Each worker runs in a separate process and can call webarena_login independently.
    
    Note: This class imports dependencies inside methods to avoid module import issues.
    """
    
    def __init__(self, worker_id: int, batch_id: int, workspace_root: str):
        """
        Initialize the Ray worker.
        
        Args:
            worker_id: Unique identifier for this worker
            batch_id: Batch ID to use for login (must be 0)
            workspace_root: Path to workspace root for imports
        """
        import sys
        from pathlib import Path
        
        # Add workspace root to Python path for imports
        if workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
        
        self.worker_id = worker_id
        self.batch_id = batch_id
        self.workspace_root = workspace_root
        self.driver = None
        print(f"[Ray Worker {self.worker_id}] Initialized with batch_id={self.batch_id}")
    
    def login(
        self,
        web_name: str,
        url: str,
        webarena_host_dict: dict,
        num_containers_per_machine: int
    ) -> Tuple[int, bool, str]:
        """
        Perform login using webarena_login function.
        
        Args:
            web_name: Web domain name (e.g., 'gitlab')
            url: Starting URL
            webarena_host_dict: WebArena host configuration as dict (for Ray serialization)
            num_containers_per_machine: Number of containers per machine
            
        Returns:
            Tuple of (worker_id, success, final_url)
        """
        try:
            import sys
            import types
            from pathlib import Path
            from selenium import webdriver
            
            # Ensure workspace root is in path
            if self.workspace_root not in sys.path:
                sys.path.insert(0, self.workspace_root)
            
            # Import here to avoid module import issues
            from agent_system.environments.env_package.webvoyager.webvoyager.utils_webarena import webarena_login
            
            # Reconstruct WebArenaHost from dict
            webarena_host = types.SimpleNamespace(**webarena_host_dict)
            
            # Create driver configuration
            options = webdriver.ChromeOptions()
            options.add_argument("--force-device-scale-factor=1")
            options.add_argument("--headless")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size=1024,768")
            
            # Create driver for this worker
            self.driver = webdriver.Chrome(options=options)
            
            print(f"[Ray Worker {self.worker_id}] Starting login with batch_id={self.batch_id}")
            
            success, url_mapping, final_url = webarena_login(
                web_name=web_name,
                url=url,
                driver_task=self.driver,
                webarena_host=webarena_host,
                batch_id=self.batch_id,
                num_containers_per_machine=num_containers_per_machine
            )
            
            # Note: Only batch_id=0 is supported, so effective_batch_id will always be 0
            effective_batch_id = self.batch_id % num_containers_per_machine
            expected_port = 8023 + effective_batch_id
            
            print(f"[Ray Worker {self.worker_id}] Login {'SUCCESS' if success else 'FAILED'}")
            print(f"  batch_id={self.batch_id} (only supported value), expected_port={expected_port}")
            if success and url_mapping:
                mapped_url = url_mapping[0][0]
                if ":" in mapped_url:
                    actual_port = mapped_url.split(":")[-1].split("/")[0]
                    print(f"  actual_port={actual_port}")
            
            return (self.worker_id, success, final_url or "")
            
        except Exception as e:
            print(f"[Ray Worker {self.worker_id}] Exception during login: {e}")
            return (self.worker_id, False, str(e))
        finally:
            if self.driver is not None:
                try:
                    self.driver.quit()
                except Exception:
                    pass
    
    def get_batch_id(self) -> int:
        """Return the batch_id for this worker."""
        return self.batch_id
    
    def get_worker_id(self) -> int:
        """Return the worker_id."""
        return self.worker_id


@pytest.mark.parametrize("num_workers", [2, 3, 4])
def test_ray_workers_concurrent_gitlab_logins(num_workers: int, ray_init):
    """
    Test multiple Ray workers logging into GitLab concurrently.
    
    This test simulates a realistic distributed scenario where:
    1. Multiple Ray workers are created (each in a separate process)
    2. Each worker calls webarena_login with batch_id=0 (only supported value)
    3. All workers login concurrently
    4. All workers use the same port (8023) since batch_id=0
    
    Args:
        num_workers: Number of Ray workers to create
    """
    # Initialize Ray if not already initialized
    # Set runtime_env to ensure Ray workers can import the module
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=min(num_workers, 4),
            runtime_env={
                "env_vars": {"PYTHONPATH": str(workspace_root)}
            }
        )
    
    web_name = git_task["web_name"]
    base_url = git_task["web"]
    url = base_url.replace(
        "ec2-98-81-119-107.compute-1.amazonaws.com",
        "ec2-3-136-229-169.us-east-2.compute.amazonaws.com"
    )
    
    webarena_host = WebArenaHost()
    # Convert to dict for Ray serialization
    webarena_host_dict = {
        'reddit': webarena_host.reddit,
        'gitlab': webarena_host.gitlab,
        'shopping_admin': webarena_host.shopping_admin,
        'shopping': webarena_host.shopping,
        'map': webarena_host.map
    }
    num_containers_per_machine = 1  # Only batch_id=0 is supported
    
    print(f"\n{'='*60}")
    print(f"Testing {num_workers} Ray workers for concurrent GitLab logins")
    print(f"{'='*60}")
    print(f"  num_workers: {num_workers}")
    print(f"  num_containers_per_machine: {num_containers_per_machine}")
    print(f"  batch_id: 0 (only supported value)")
    print(f"  Expected port: 8023 (for GitLab with batch_id=0)")
    
    # Create Ray workers - all use batch_id=0
    # Pass workspace_root so workers can set up their Python path
    workers = []
    for worker_id in range(num_workers):
        batch_id = 0  # Only batch_id=0 is supported
        worker = WebArenaLoginWorker.remote(
            worker_id=worker_id,
            batch_id=batch_id,
            workspace_root=str(workspace_root)
        )
        workers.append(worker)
    
    print(f"\nCreated {len(workers)} Ray workers")
    
    # Submit login tasks to all workers concurrently
    futures = []
    for worker in workers:
        future = worker.login.remote(
            web_name=web_name,
            url=url,
            webarena_host_dict=webarena_host_dict,
            num_containers_per_machine=num_containers_per_machine
        )
        futures.append(future)
    
    print(f"Submitted {len(futures)} login tasks to Ray workers")
    
    # Collect results
    results = []
    try:
        # Wait for all tasks to complete with timeout
        ready, not_ready = ray.wait(futures, num_returns=len(futures), timeout=300)
        if len(ready) < len(futures):
            print(f"Warning: Only {len(ready)}/{len(futures)} tasks completed within timeout")
        
        for future in ready:
            try:
                result = ray.get(future, timeout=10)
                results.append(result)
            except Exception as e:
                print(f"Error getting result: {e}")
                results.append((None, False, str(e)))
    except Exception as e:
        print(f"Error collecting results: {e}")
        # Try to get whatever results we can
        for future in futures:
            try:
                result = ray.get(future, timeout=1)
                results.append(result)
            except Exception:
                pass
    
    # Verify results
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"{'='*60}")
    
    success_count = 0
    worker_batch_ids = {}
    
    for worker_id, success, final_url in sorted(results):
        if worker_id is not None:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"Ray Worker {worker_id}: {status}")
            if success:
                success_count += 1
                # Get the batch_id used by this worker
                try:
                    batch_id = ray.get(workers[worker_id].get_batch_id.remote())
                    worker_batch_ids[worker_id] = batch_id
                except Exception:
                    pass
    
    print(f"\nTotal: {success_count}/{num_workers} successful logins")
    
    # Assertions
    assert len(results) == num_workers, (
        f"Expected {num_workers} results, got {len(results)}"
    )
    assert success_count == num_workers, (
        f"Expected all {num_workers} logins to succeed, but only {success_count} succeeded. "
        f"Failed worker IDs: {[wid for wid, s, _ in results if not s]}"
    )
    
    # Verify all workers used batch_id=0
    if len(worker_batch_ids) == num_workers:
        batch_ids_used = set(worker_batch_ids.values())
        assert batch_ids_used == {0}, (
            f"Expected all workers to use batch_id=0, but got: {batch_ids_used}"
        )
        print(f"\n✓ All workers used batch_id=0 (only supported value)")
    
    print(f"\n✓ All {num_workers} Ray workers successfully completed concurrent logins!")
    
    # Cleanup: terminate workers
    for worker in workers:
        try:
            ray.kill(worker)
        except Exception:
            pass


@pytest.fixture(scope="module")
def ray_init():
    """Fixture to initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    yield
    # Cleanup: shutdown Ray after all tests
    if ray.is_initialized():
        ray.shutdown()

