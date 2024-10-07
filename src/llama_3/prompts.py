custom_prompt = {"system":"You are an expert radiologist. Below here is a patient's radiology report containing a recommendation and a possible candidate that satisfies the recommendation.",
                 "user": ''' Does the candidate satisfy the recommendation? Answer in the following format. 'YES' if the candidate satisfies the recommendation else 'NO'.
            Output format: <answer start> your answer (YES or NO) <answer end> '''}
gpt_base_prompt = '''You are a board certified radiologist. You will compare Report A and Report B. Your goal is to check whether Report B is a proper follow-up of Report A. Answer in the following format. 'True' if Report B is a proper follow-up else 'False'.
                Output format: <answer start> your answer (True or False) <answer end>'''
gpt_adv_prompt = '''You are a board certified radiologist. You will compare Report A and Report B mostly focussing on information from the sentence in Report A which explicitly suggests a follow-up examination. Your goal is to check whether Report B is a proper follow-up of Report A. While a proper follow-up does not always have to use the same imaging test, same day evaluations are not considered as a correct follow-up.
                Note: Modality types do not need to match the recommended imaging test, if it can still qualify as a substitute.
                Answer in the following format. 'True' if Report B is a proper follow-up else 'False'.
                Output format: <answer start> your answer (True or False) <answer end>'''