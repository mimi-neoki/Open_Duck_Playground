# joystick_chatgpt.py などに切り出しておくと管理しやすい
import openai, os, json, numpy as np
from pydantic import BaseModel, Field, ValidationError

COMMAND_RANGES = {
    "lin_x":     (-0.15,  0.15),
    "lin_y":     (-0.20,  0.20),
    "ang_theta": (-1.00,  1.00),
    "neck_pitch":(-0.34,  1.10),
    "head_pitch":(-0.78,  0.78),
    "head_yaw":  (-1.50,  1.50),
    "head_roll": (-0.50,  0.50),
}

class JoystickCmd(BaseModel):
    lin_x:      float = Field(..., ge=-0.15, le=0.15)
    lin_y:      float = Field(..., ge=-0.20, le=0.20)
    ang_theta:  float = Field(..., ge=-1.0,  le=1.0)
    neck_pitch: float = Field(..., ge=-0.34, le=1.10)
    head_pitch: float = Field(..., ge=-0.78, le=0.78)
    head_yaw:   float = Field(..., ge=-1.50, le=1.50)
    head_roll:  float = Field(..., ge=-0.50, le=0.50)

class ChatGPTJoystick:
    def __init__(self, model="gpt-4o-mini"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.sys_prompt = (
          """You are a joystick command generator for a MUJOCO humanoid.

          Coordinate frame & units
          ------------------------
          • lin_x  (m/s)  : +x is **forward** relative to robot torso
          • lin_y  (m/s)  : +y is **left**   relative to robot torso
          • ang_theta (rad/s): +θ is **CCW (left) yaw** around vertical axis
          • +θ  : counter-clockwise (left) yaw
          • −θ  : clockwise       (right) yaw
          • neck_pitch : (rad): +pitch is **up** neck relative to robot torso
          • head_pitch : (rad): +pitch is **up** head relative to robot torso
          • head_yaw : (rad): +yaw is **CCW (left) yaw** head around vertical axis
          • head_roll (rad) : +roll is **CCW (left) roll** head around horizontal axis 
           


          Limits
          ------
          lin_x  [-0.15, 0.15]
          lin_y  [-0.20, 0.20]
          ang_theta [-1.0, 1.0]
          neck_pitch [-0.34, 1.10]
          head_pitch [-0.78, 0.78]
          head_yaw  [-1.50, 1.50]
          head_roll [-0.50, 0.50]

          Mapping examples
          ----------------
            "turn left"  → {"ang_theta": +0.8}
            "turn right" → {"ang_theta": -0.8}
            "head up" → {"head_pitch": +0.7}, {neck_pitch: +1.0}

          Output specification
          --------------------
          Always return *only* valid JSON that matches the function schema,
          with numeric values inside the above ranges. Do not add text.

          """
        )
        self.fn_spec = {
            "name": "set_joystick",
            "description": "Return one time-step worth of joystick commands",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "number"} for k in JoystickCmd.model_fields},
                "required": list(JoystickCmd.model_fields),
            },
        }

    def ask(self, user_prompt: str) -> np.ndarray:
        """Natural-language → 7-dim command vector (np.float32)."""
        rsp = openai.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            functions=[self.fn_spec],
            function_call={"name": "set_joystick"},  # enforce JSON
        )
        fn_args = rsp.choices[0].message.function_call.arguments
        try:
            cmd = JoystickCmd(**json.loads(fn_args))
        except (ValidationError, json.JSONDecodeError):
            # 万一壊れたらゼロコマンドを返す
            return np.zeros(7, dtype=np.float32)
        return np.array(list(cmd.model_dump().values()), dtype=np.float32)
