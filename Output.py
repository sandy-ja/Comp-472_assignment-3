#importing io library
import io

#creating the trace files 
def Trace_file(name, id_test, y_predict, y_test, score):
    #name the file
    fileName = ("trace_NB-BOW-"+ name +".txt")
    output = open(fileName,"wt")
    for i in range(0, len(id_test)):
        output.write(str(id_test[i]))
        output.write("  ")
        output.write(str(y_predict[i]))
        output.write("  ")
        output.write(str(score[i]))
        output.write("  ")
        output.write(str(y_test[i]))
        output.write("  ")
        if y_predict[i] == y_test[i]:
            output.write("correct\r")
        else:
            output.write("wrong\r")

#creating the evaluation files 
def Eval_file(name, accuracy, per_yes, per_no, rec_yes, rec_no, f_yes, f_no):
    #name the file
    fileName = ("eval_NB-BOW-"+ name +".txt")
    output = open(fileName,"wt")
    output.write(str(accuracy)+"\r")
    output.write("\n")
    output.write(str(per_yes))
    output.write("  ")
    output.write(str(per_no)+"\r")
    output.write("\n")
    output.write(str(rec_yes))
    output.write("  ")
    output.write(str(rec_no)+"\r")
    output.write("\n")
    output.write(str(f_yes))
    output.write("  ")
    output.write(str(f_no)+"\r")
