from pathlib import Path
from repo_manager import RepoManager  # Adjust import based on your file name

# Initialize the manager
manager = RepoManager(repos_dir=Path("."))

# Clone the GeoMLAMA repository
repo_path = manager.clone_or_update(
    repo_url="https://github.com/WadeYin9712/GeoMLAMA.git",
    repo_name="GeoMLAMA"
)

# Optionally install dependencies
manager.setup_repo(repo_path, install_deps=True)

print(f"Repository cloned to: {repo_path}")