import shutil
import os
from git import Repo
import filecmp
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# 设置信号处理器
signal.signal(signal.SIGALRM, timeout_handler)

def add_file_to_repo(file_path, repo_path='/home/jqxu/Ragas/datasets', branch='master', repo_url='https://github.com/CazeroZ/Images'):
    try:
        signal.alarm(10)  # 设置10秒超时
        repo = Repo(repo_path)
        repo.git.checkout(branch)

        file_name = os.path.basename(file_path)
        dest_path = os.path.join(repo_path, file_name)
        repo_file_path = os.path.join(repo_path, file_name)

        if os.path.exists(repo_file_path):
            if filecmp.cmp(file_path, repo_file_path, shallow=False):
                origin = repo.remote(name='origin')
                origin.fetch()
                try:
                    origin_file_status = origin.repo.git.ls_tree(branch, repo_file_path)
                    if origin_file_status:
                        return f"{repo_url}/raw/{branch}/{file_name}"
                except Exception as e:
                    print(f"Error checking remote file status: {e}")

        shutil.copyfile(file_path, dest_path)
        repo.index.add([file_name])
        repo.index.commit(f'Add or update file: {file_name}')
        origin = repo.remote(name='origin')
        origin.push(branch)
        return f"{repo_url}/raw/{branch}/{file_name}"

    except TimeoutException:
        print("Operation timed out, trying again...")
        return add_file_to_repo(file_path, repo_path, branch, repo_url)
    finally:
        signal.alarm(0)  # 取消超时警报

def test_module():
    print("test the module git_image")

def get_file_url(file_path, repo_path='/home/jqxu/Ragas/datasets', branch='master', repo_url='https://github.com/CazeroZ/Images'):
    return add_file_to_repo(file_path, repo_path, branch, repo_url)

def delete_file_from_repo(file_name, repo_path='/home/jqxu/Ragas/datasets', branch='master', repo_url='https://github.com/CazeroZ/Images'):
    repo = Repo(repo_path)
    repo.git.checkout(branch)  

    file_path = os.path.join(repo_path, file_name)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    repo.index.remove([file_path])
    repo.index.commit('Delete file: ' + file_name)
    origin = repo.remote(name='origin')
    origin.push(branch)  
'''
image_pth="/home/jqxu/Ragas/datasets/figure09-01-05.jpg"
url=get_file_url(image_pth)
print(url)
delete_file_from_repo(os.path.basename(image_pth))
'''