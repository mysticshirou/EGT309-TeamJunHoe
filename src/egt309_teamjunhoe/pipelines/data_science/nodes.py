from interfaces import Model

def model_choice(params):
    if params.get("decision_tree_model", None) == True:
        class ChosenModel(Model):
            

    return ChosenModel