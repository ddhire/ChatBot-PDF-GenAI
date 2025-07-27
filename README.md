# ChatBot-PDF-GenAI
ChatBot for getting information from uploaded PDF

# install python 3.10 in mac
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew install python@3.10
python3.10 --version
# Create conda environamnet
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-*.sh
source ~/.zshrc   # or ~/.bash_profile if using bash
conda --version
conda create -n Chatbot python=3.10 -y
conda activate Chatbot
pip install cryptography --upgrade