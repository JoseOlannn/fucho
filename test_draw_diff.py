import pandas as pd
import subprocess

def test_thresholds():
    original_file = ""
    with open("model.py", "r") as f:
        original_file = f.read()
        
    thresholds = [0.05, 0.10, 0.15, 0.20]
    
    old_logic = """    if final_prob_w1 > final_prob_d and final_prob_w1 > final_prob_w2:
        expected_class = "H"
    elif final_prob_w2 > final_prob_d and final_prob_w2 > final_prob_w1:
        expected_class = "A"
    else:
        expected_class = "D\""""
        
    for t in thresholds:
        new_logic = f"""    if abs(final_prob_w1 - final_prob_w2) < {t}:
        expected_class = "D"
    elif final_prob_w1 > final_prob_w2:
        expected_class = "H"
    else:
        expected_class = "A\""""
        
        new_file = original_file.replace(old_logic, new_logic)
        with open("model.py", "w") as f:
            f.write(new_file)
            
        res = subprocess.run(["venv/bin/python", "test_tab6.py"], capture_output=True, text=True)
        try:
            output = res.stdout.strip()
            lines = output.split('\n')
            brier_line = [l for l in lines if 'Brier' in l][0]
            ganador_line = [l for l in lines if 'Ganador' in l][0]
            exacto_line = [l for l in lines if 'Exacto' in l][0]
            
            brier = float(brier_line.split(': ')[1])
            ganador_pct = float(ganador_line.split(': ')[1].split('%')[0])
            exacto_pct = float(exacto_line.split(': ')[1].split('%')[0])
            print(f"Diff Umbral {t}: Ganador {ganador_pct:.1f}%, Exacto {exacto_pct:.1f}%")
        except Exception as e:
            pass
            
    with open("model.py", "w") as f:
        f.write(original_file)

if __name__ == '__main__':
    test_thresholds()
