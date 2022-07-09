"""
csv files from /outputcsv
to
txt files /output
"""
import pandas as pd

# filename_out2 = "brute"
# filename_out2 = "approx"
filename_out2 = "random"

dataname = 'd_sentiment'
#
# method = 'DS'
# method = 'MV'
# method = 'GLAD'
# method = 'BWA'
# method = 'PM'
method = 'EBCC'
# method = 'BCC'
# method = 'ZC'

for i in range(9, 10):
    Pr = (i * 1.0) / 10
    for k in range(2, 4):

        filename2 = "./outputcsv/" + dataname + "/" + dataname +"_"+str(Pr)+"acc+"+filename_out2+"(k="+str(k)+")"+method+"_s.csv"
        # filename2 = "./outputcsv/" + dataname + "/" + dataname +"_"+str(Pr)+"acc+"+filename_out2+"(k=6)"+method+"_low_0.8_0.9_k="+str(k) + ".csv"
        # filename3 = "./outputcsv/" + dataname + "/" + dataname +"_"+str(Pr)+"acc+"+filename_out3+"(k="+str(k) + ")_s.csv"

        # x1 = pd.read_csv(filename1)
        # with open("./output/" + dataname + '_' + str(Pr) + "acc+"+ filename_out1 + "(k=" + str(k) + ")_3.txt", "w") as f:
        #     id = 0
        #     step = 40
        #     for i in range(0, len(x1), step):
        #         f.write(str(id))
        #         f.write(' ')
        #         id += 40*3*2
        #         f.write(str(-x1.iloc[i, 1]))
        #         f.write('\n')
        #     f.write(str((len(x1)-1)*3*2)+' ')
        #     f.write(str(-x1.iloc[len(x1)-1, 1]))

        x2 = pd.read_csv(filename2)
        # print(x2)
        # with open("./output/" + dataname + '_' + str(Pr) + "acc+"+ filename_out2 + "(k=2)"+ method +"_low_0.8_0.9_k="+str(k) + ".txt", "w") as f:
        with open("./output/" + dataname + '_' + str(Pr) + "acc+" + filename_out2 + "(k=" + str(k) + ")_s.txt", "w") as f:
            id = 0
            step = 60 // k
            for i in range(0,len(x2),step):
                f.write(str(id))
                f.write(' ')
                id += step * 2 * k
                f.write(str(-x2.iloc[i, 1]))
                f.write('\n')
            f.write(str(2004) + ' ' + str(-x2.iloc[len(x2) - 1, 1]))
            # id = 0
            # step = 40
            # for i in range(0, len(x2), step):
            #     f.write(str(id))
            #     f.write(' ')
            #     id += k * step*2
            #     f.write(str(-x2.iloc[i, 1]))
            #     f.write('\n')
            # # f.write(str(x2.iloc[len(x2)-1, 0]*k*2)+' '+str(-x2.iloc[len(x2)-1, 1]))
            # f.write(str(2004) + ' ' + str(-x2.iloc[len(x2) - 1, 1]))


        # x3 = pd.read_csv(filename3)
        # with open("./output/" + dataname + '_' + str(Pr) + "acc+"+ filename_out3 + "(k=" + str(k) + ")_3.txt", "w") as f:
        #     id = 0
        #     step = 40
        #     for i in range(0, len(x3), step):
        #         f.write(str(id))
        #         f.write(' ')
        #         id += 40 * 3 * 2
        #         f.write(str(-x3.iloc[i, 1]))
        #         f.write('\n')
        #     f.write(str((len(x3)-1) * 3 * 2) + ' ')
        #     f.write(str(-x3.iloc[len(x3) - 1, 1]))