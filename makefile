VENV=.venv/bin
APP=app.py
ROBOT=robot.py

all: 
	@$(VENV)/flask --app $(APP) run
	