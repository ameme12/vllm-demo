import git
import shutil
from pathlib import Path
from typing import Dict
import subprocess

class RepoManager:
    """Manages cloning and updating GitHub repositories"""
    
    def __init__(self, repos_dir: Path = Path("./repos")):
        self.repos_dir = repos_dir
        self.repos_dir.mkdir(exist_ok=True, parents=True)
        
    def clone_or_update(self, repo_url: str, repo_name: str = None, branch: str = None) -> Path:
        """Clone repository or update if already exists"""
        if repo_name is None:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            
        repo_path = self.repos_dir / repo_name
        
        if repo_path.exists():
            print(f"Updating {repo_name}...")
            try:
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
                if branch:
                    repo.git.checkout(branch)
            except Exception as e:
                print(f"Error updating {repo_name}: {e}")
        else:
            print(f"Cloning {repo_name}...")
            clone_args = {'to_path': repo_path}
            if branch:
                clone_args['branch'] = branch
            git.Repo.clone_from(repo_url, **clone_args)
            
        return repo_path
    
    def setup_repo(self, repo_path: Path, install_deps: bool = True):
        """Install dependencies for repository"""
        if not install_deps:
            return
            
        requirements_file = repo_path / "requirements.txt"
        if requirements_file.exists():
            print(f"Installing dependencies from {requirements_file}")
            subprocess.run(['pip', 'install', '-r', str(requirements_file)], check=True)