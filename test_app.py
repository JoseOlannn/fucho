import model
result = model.predict_match('France', 'Argentina')
print("poss1:", result.get('poss1'))
print("poss2:", result.get('poss2'))
