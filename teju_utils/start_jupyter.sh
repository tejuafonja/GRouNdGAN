mkdir -p ~/.jupyter
echo "import os
c.NotebookApp.allow_root = True
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = int(os.getenv('JUPYTER_PORT','5555'))
c.NotebookApp.custom_display_url = 'http://hostname:%d' % (c.NotebookApp.port)
c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
pip install jupyter
python3.9 -m pip install ipykernel
python3.9 -m pip install jupyter
python3.9 -m ipykernel install --user
python3.9 -m ipykernel install --user --name=python3.9 --display-name "Python (python3.9)"
jupyter lab