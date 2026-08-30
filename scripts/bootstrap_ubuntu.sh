#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the operating system." >&2
  exit 2
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "Expected Ubuntu 22.04; found ${PRETTY_NAME:-unknown}." >&2
  exit 2
fi

for command_name in python3 vcs rosdep colcon; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
done

if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble is not installed at /opt/ros/humble." >&2
  exit 2
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
mkdir -p "${repo_root}/src/vendor"
vcs import --recursive "${repo_root}/src/vendor" < "${repo_root}/vendor/robosense.repos"
git -C "${repo_root}/src/vendor/rslidar_sdk" submodule update --init --recursive

rosdep install \
  --from-paths "${repo_root}/src" \
  --ignore-src \
  --rosdistro humble \
  -r -y

echo "Dependencies are ready. Build with:"
echo "  colcon build --symlink-install --cmake-args -DENABLE_IMU_DATA_PARSE=ON -DENABLE_TRANSFORM=ON"
