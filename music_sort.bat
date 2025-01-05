:: filepath: /c:/Users/sukhi/Documents/Workspace/CodingWorkspace/Music sorter/run_music_sorter.bat
@echo off
:: Activate the conda environment named 'langchain'
call conda activate langchain

:: Run the Python script
python music_sorter.py

:: Deactivate the conda environment
call conda deactivate