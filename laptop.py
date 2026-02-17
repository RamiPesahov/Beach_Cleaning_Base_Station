from flask import Blueprint, jsonify, request, requests

ROBOT_URL = "http://127.0.0.1:5000/robot/getcommands"

laptop_bp = Blueprint('laptop', __name__)

@laptop_bp.route('/capture')
def take_photo():
    return jsonify({"message": "Photo captured on laptop"})

def send_robot_command(action, value=None):
    
        # function that makes the robot move

    payload = {
        "action": action,
        "value": value
    }
    
    try:
        response = requests.post(ROBOT_URL, json=payload)
        
        if response.status_code == 200:
            print(f"Command Succsfull:{action}")
            print(f"Current Robot status: {response.json().get('robot_state')}")

        else:
            print(f"The server made an error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to the Robot")

def move_forward(distance):
    send_robot_command("MOVE_FORWARD", value=distance)

def stop_robot():
    send_robot_command("STOP")

if __name__ == "__main__":
    move_forward(10)
    stop_robot()
