from flask import Blueprint, jsonify, request

robot_bp = Blueprint('robot', __name__)

# Virtual Robot for now
virtual_robot = {
    "x": 0,
    "y": 0,
    "status": "IDLE",
    "battery": 100
}

@robot_bp.route('/move', methods=['POST'])
def move_robot():
    return jsonify({"status": "Robot is moving"})

@robot_bp.route('/status')
def robot_status():
    return jsonify({"battery": "85%", "connected": True})


@robot_bp.route('/getcommands', methods=['POST'])
def handle_command():
    data = request.json
    action = data.get('action')
    
    # movement instructions
    if action == "MOVE_FORWARD":
        virtual_robot["y"] += 1
        virtual_robot["status"] = "MOVING"
    elif action == "STOP":
        virtual_robot["status"] = "IDLE"
    
    print(f"--- VIRTUAL ROBOT UPDATE ---")
    print(f"Current Position: ({virtual_robot['x']}, {virtual_robot['y']})")
    print(f"Current Status: {virtual_robot['status']}")
    
    return jsonify({
        "status": "ACK",
        "robot_state": virtual_robot
    })

@robot_bp.route('/state', methods=['GET'])
def get_state():
    return jsonify(virtual_robot)