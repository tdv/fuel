# fuel - is the fuel to fake your webcam background under Linux. 

**Note:** 'fuel' aims for Intel GPUs (mostly for cases where CPUs have integrated graphics). 

# Introduction
'Fuel' is a webcam background removal solution with budget-friendly GPU optimization via OpenVINO for integrated GPUs. It is well-suited for systems lacking an NVIDIA GPU and equipped with Intel CPUs featuring integrated graphics. The solution can be used as a fake webcam under Linux for online meetings, providing a desirable artificial background.

# Features 
- background substitution
- blurring background

# Major dependencies 
- OpenVINO
- OpenCV

# OS 
The solution has been tested under Ubuntu 23.10 and might be to work on other platforms, but hasn't been tested.

# Language 
- Python

# Install  
Installation might not be straightforward and could require some system administration knowledge.
- First of all, you need to [check](https://ark.intel.com/content/www/us/en/ark.html#@PanelLabel122139) if your CPU is on the list of OpenVINO supported CPUs 
- [Install Intel OpenVINO Toolkit for Linux](https://docs.openvino.ai/2023.3/openvino_docs_install_guides_installing_openvino_apt.html) 
- Install `v4l2loopback` and create a virtual webcam. It might be performed like ```sudo apt install -y v4l2loopback-dkms v4l-utils v4l2loopback-utils 
sudo v4l2loopback-ctl add --exclusive-caps=1 --name="fake-cam" /dev/video3
v4l2-ctl --list-devices``` or quite close to.
- Install Python 3 if you haven't done so already. (In my case, everything was tested on Python 3.11) 
- Install a virtual environment for Python 3 (venv)
- Clone repository  ```git clone https://github.com/tdv/fuel.git``` 
- Go to the directory ```cd fuel``` 
- Make a virtual environment using the command ```python3 -m venv venv```
- Run the virtual environment using ```source ./venv/bin/activate``` 
- Install all required packages using ```pip3 install -r requirements.txt``` 

**Congrats, you are ready to use this!**  

# How to Use 
- Activate the virtual environment (if not already active) 
- Explore available commands via ```python3 fuel.py -h``` 
- Start using, for example, with ```python3 fuel.py -s webcam -w /dev/video0 -d fakecam -m background -b ./background/bg1.jpg -f /dev/video3``` 
- Test it, for instance, with ```ffplay -f v4l2 /dev/video3``` 
