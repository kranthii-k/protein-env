from models import ProteinObservation, StepResult, RewardBreakdown, StepInfo, VariantInfo, TaskType

class ProteinEnvClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.step_idx = 0
        self.task_type = TaskType.EASY

    def reset(self, task_type):
        self.task_type = TaskType(task_type)
        self.step_idx = 0
        return ProteinObservation(
            protein_id="P12345",
            sequence="MOCKSEQUENCE",
            sequence_length=12,
            task_type=self.task_type,
            task_description=f"Mock task description for {task_type}",
            available_tools=["get_esm2_embedding"],
            step_number=self.step_idx,
            max_steps=10
        )

    def step(self, action):
        self.step_idx += 1
        # Let's say it finishes after 1 step of prediction for testing
        done = True
        reward = 1.0 if self.task_type == TaskType.EASY else 0.8
        
        obs = ProteinObservation(
            protein_id="P12345",
            sequence="MOCKSEQUENCE",
            sequence_length=12,
            task_type=self.task_type,
            task_description=f"Mock task description for {self.task_type.value}",
            available_tools=["get_esm2_embedding"],
            step_number=self.step_idx,
            max_steps=10
        )
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=StepInfo(reward_breakdown=RewardBreakdown(base_score=reward))
        )
