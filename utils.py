
def check_valid_question(question):
    option = 'appropriate option' in question.split('.')[-1].lower()
    return not option
