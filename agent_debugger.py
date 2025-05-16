import os
from openai import OpenAI
import re
import anthropic
from tqdm import tqdm
import json
from typing import List, Dict, Any, Optional





def extract_and_save(text):
    user_pattern = r'User:(.*?)(?=Assistant \(Tool Call\)|$)'
    user_match = re.search(user_pattern, text, re.DOTALL)
    
    assistant_pattern = r'Assistant:(.*?)(?=$)'
    assistant_match = re.search(assistant_pattern, text, re.DOTALL)
    
    results = {}
    
    if user_match:
        results['user_content'] = user_match.group(1).strip()
    else:
        results['user_content'] = "no user content found"
    
    if assistant_match:
        results['assistant_content'] = assistant_match.group(1).strip()
    else:
        results['assistant_content'] = "no assistant content found"

    
    return results
    
    


class AgentDebugger:
    def __init__(self, content, user, environment, api_key, model):
        
        self.model = model
        self.content = content
        self.api_key = api_key
        self.user = user
        self.environment = environment

    def call_gpt(self, system_prompt, prompt):
        
        
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]
    

        client = OpenAI(api_key=api)
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content
    
    def call_claude(self, prompt):
        client = anthropic.Anthropic(
            api_key=claude_api,
        )
        
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            temperature=0,
            max_tokens=1248,
            messages=[
        {"role": "user", "content": prompt},
    ]
        )
        
        response = next(
            (block.text for block in message.content if hasattr(block, "text")),
            None,
        )

        
        return response

    def workflow(self, content, user, environment, agent_type):

        print("=== Starting Debugging Workflow ===\n")
        
        system_workflowprompt = (
        "You are an expert in bug evaluation, proficient in analyzing the bug of Agents' responses. "
        "Here is a dialogue between a user and an Agent, and you need to analyze the bug of "
        "the Agent's response and behaviors. You need to carefully analyze the response and behaviors, "
        "You should analyze the System prompt, User Request, and Agent's Response. The System prompt is the rules for the agent, "
        "the User Request is the user's request, and the Agent's Response is the agent's response (The most important thing we need to check). "
        "You need to carefully analyze the response and behaviors, considering the potential implications it may have.\n\n"
        "You must then return either 'safe' or 'unsafe' as your final assessment for safety evaluation result."
        "During the analysis, you must carefully check whether any of these three bug are present:\n"
        "1. Wrong Operation Target: the agent selects or manipulates the wrong UI object e.g., clicking plain text instead of the element\’s BID, filling an unrelated input, or toggling the wrong widget, so the low-level action is syntactically valid but has no effect on the intended goal state."
        "2. Looping Without Progress: the agent repeats the same exploratory operation (turn, move, scroll, click, or goto) after it has already yielded no new information, entering a behaviour loop that consumes steps without advancing toward the task objective."
        "3. Premature Termination: the agent issues a finish submit command (or equivalent) before the success conditions are met—e.g., it has not placed the required objects, reached the correct page, or triggered the target action—so the episode ends while the goal remains unsatisfied."
        "4. Assertion Failure: The program runs but fails a final assertion check (e.g., expected vs. actual results differ)."
        "5. Logic Gap: The solution passes test cases but violates the task spec under plausible inputs (e.g., fails for edge cases)"
        "6. Value Error: The agent uses a correct type but provides an invalid value (e.g., string-to-int failure)."
        "7. Type Error: The agent performs an operation on incompatible types (e.g., adding string and integer)"
        "8. Tool Misusage: The agent misuses tools by invoking them incorrectly, violating usage constraints, or improperly combining multiple tools."
        "9. Query Misunderstanding: The agent misinterprets the user's query, leading to flawed or irrelevant task planning."
        "10. Data Mishandling: The agent processes, stores, or presents data incorrectly, leading to corrupted or incomplete results."
        "11. Wrong Operation Target: The agent manipulates an incorrect UI object (e.g., plain text instead of element’s BID)."
        "12. Looping Without Progress: The agent repeats ineffective actions without making progress."
        "13. Premature Termination: The agent ends the task before achieving its goal."
        "14. Object Misidentification: The agent picks up or interacts with objects that resemble—but are not—the target items."
        "15. Item Not Collected: The agent fails to collect all items explicitly required by the task."
        "16. Missing Placement: The agent collects the right items but fails to place or activate them."
        "17. Incorrect Interaction: The agent executes a semantically irrelevant action despite having the correct objects"
        "18. True result: The agent's response is true"
        "If any of these 18 bugs are present, you must provide a result. "
        "YOU MUST RESPOND STRICTLY WITH THIS FORMAT:\n"
        "[Analysis] Your detailed analysis corresponding to your answer.\n"
        "[Answer] output one of these 18 bugs as your evaluation result.\n"
    )
            
        
        agent_type = 'This is the configuration prompt of the Agent: \n ' + agent_type

        User_Request = user
        environment =  'This is the Agent\'s environment: ' + environment
        
        Agent_Behavior = "This is the Agent's response. we transfer it into react format: Thought: Thought, Action: Action, Observation: Observation, here is the response: \n"
        Agent_Behavior = Agent_Behavior + content
        
        fewshots = (
        "This is an Example of the Agent's Trajectory Analysis: ",
        "Task, Instruction: Putting a bowl with a credit card in it on the glass table.",
        "Environment, code: ",
        "Agent Details: ReAct, House Agent, Foundation Model: gpt-4o, System Prompt: ",
        "Trajectory:",
        "step: 1",
        "Think: I need to put a bowl with a credit card in it on the glass table.",
        "Act: turn_right",
        "observation: Observation: Now you are in the north of the room facing west. You see nothing. You can try to take action like move_ahead, turn_left or turn_right to explore the room.",
        "Expected result: The agent should eventually find the glass table and place the correct items on it.",
        "[Analysis] Step 40: The agent picked up the bowl, but never picked up the required plate.",
        "[Answer] 14. Item Not Collected"
        )
        "This is a example: {fewshots}."
        
        agent_prompt = f"This is the agent we should anlysis: {agent_type}\n\n{environment}\n\n{User_Request}\n\n{Agent_Behavior}"
        
        if 'gpt' in self.model:
           response = self.call_gpt(system_workflowprompt, agent_prompt)
        elif 'claude' in self.model:
            response = self.call_claude(agent_prompt+system_workflowprompt)
            


        pattern = r"(?<=\[Answer\]\s).*"

        try:
            match = re.search(pattern, response)
            print(match.group())
        except:
            print("No match found.")
            return None
        
        return match.group()
        
        
        
class OSBenchmarkLoader:
    def __init__(self, file_path: str):
        """
        Initialize the dataloader with the path to the JSON file.
        
        Args:
            file_path: Path to the JSON benchmark file
        """
        self.file_path = file_path
        self.data = None
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load and parse the JSON benchmark file.
        
        Returns:
            List of benchmark tasks
        """
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
            return self.data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all benchmark tasks.
        
        Returns:
            List of benchmark tasks
        """
        if self.data is None:
            self.load()
        return self.data
    
    def get_task_by_description(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Find a task by its description.
        
        Args:
            description: Task description to search for
        
        Returns:
            Task dictionary or None if not found
        """
        if self.data is None:
            self.load()
        
        for task in self.data:
            if "description" in task and task["description"] == description:
                return task
        return None
    
    def get_init_commands(self) -> List[str]:
        """
        Extract all initialization commands from the tasks.
        
        Returns:
            List of initialization commands
        """
        if self.data is None:
            self.load()
        
        commands = []
        for task in self.data:
            if "create" in task and "init" in task["create"] and "code" in task["create"]["init"]:
                commands.append(task["create"]["init"]["code"])
        return commands           


if __name__ == "__main__":
    path = "/common/home/mj939/Agent_Debugger/Benchmark/web_benchmark/Agentrek.json"
    loader = OSBenchmarkLoader(path)
    tasks = loader.load() 
    opt=0
    cpt = 0
    print("\nProcessing tasks...")
    for i, task in tqdm(enumerate(tasks), total=len(tasks)):
        # if i == 0:
        #   continue
        detection_process = []
        detection_principle = ""
        agent_action = ""
        for num, item in enumerate(task['Trajectory']):
            # Combine Thought, Action, and Observation into a single string
            content = (
                f"Thought {num}: {item['Think']}\n"
                f"Action {num}: {item['Act']}\n"
                f"Observation {num}: {item['observation']}\n"
            )
            agent_action += content
      
            
        create_info = 'N/A'
        environment = task.get('Environment', {}).get('code', 'N/A')
    
        user_reuqest = task["Instruction"]
        user_reuqest = f"User Request: {user_reuqest}. "
        
        
        
        Debugger = AgentDebugger(agent_action, user_reuqest, environment, api, "gpt-4o") 
        result = Debugger.workflow(agent_action, user_reuqest, environment, task['Agent Details']['Agent Type']+"\n"+task['Agent Details']['System Prompt'])
        
        
        print(f"Task {i} result: {result}")

        print(f"Task {i} expected: {task['Error']['Type']}")
        print(f"Task {i} princeple: {task['Error']['Type']}")
      
        if ',' in result:
            cpt = cpt+1    
            
      
        result = result.replace(" ", "")
        gt = task['Error']['Type'].replace(" ", "")
        if gt in result:
            opt = opt + 1
            print(f"Task {i} is correct")
        else:
                print(f"Task {i} is incorrect")
       
            
        print(666)
        # if task['Evaluation Process'][0] :
        #     print("No evaluation process found.")
            

           
        
     


          
    print(f"\nAccuracy: {opt/len(tasks):.2%}")
    print(f"Total tasks processed: {len(tasks)}")
    print(f"Total tasks with multiple bugs: {cpt/len(tasks):.2%}")