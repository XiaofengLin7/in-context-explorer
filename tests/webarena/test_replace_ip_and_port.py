import sys
import os
from pathlib import Path

# Add workspace root to Python path for imports
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from agent_system.environments.env_package.webvoyager.webvoyager.utils_eval import (
    replace_ip_and_port,
)


def test_replace_ip_and_port_with_ipv4():
    target_url = "http://10.0.0.5:7770"
    url_to_modify = "http://WEBARENA_HOST:PORT/rest/default/V1/orders"

    result = replace_ip_and_port(target_url, url_to_modify)

    assert result == "http://10.0.0.5:7770/rest/default/V1/orders"


def test_replace_ip_and_port_with_hostname():
    target_url = "http://ec2-3-136-229-169.us-east-2.compute.amazonaws.com:9999"
    url_to_modify = "http://WEBARENA_HOST:PORT/rest/default/V1/orders"

    result = replace_ip_and_port(target_url, url_to_modify)

    assert (
        result
        == "http://ec2-3-136-229-169.us-east-2.compute.amazonaws.com:9999/rest/default/V1/orders"
    )


def test_replace_ip_and_port_without_match_returns_original():
    target_url = "https://no-port-hostname"
    url_to_modify = "http://WEBARENA_HOST:PORT/rest/default/V1/orders"

    result = replace_ip_and_port(target_url, url_to_modify)

    assert result == url_to_modify

