import numpy as np
import random
import client as server
import copy
data = [0.0, -1.1025519195689977e-12, -2.2928750879751944e-13, 6.750529375825805e-11, -2.361782935623884e-10, -1.6415897057292425e-15, 7.887826319348786e-16, 1.8992112359495576e-05, -1.2697818806537515e-06, -6.92838031014448e-09, 4.716028062849655e-10]


# data = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
mutate_range = 0.1
population = [list(data) for i in range(10)]
offsprings = []
popfit = []
popfit2 = []
mom = []
dad = []
train_factor = 0.5
# population[0] = [0.0, -1.28344644319375e-12, -2.794777419645668e-13, 5.1859378234020296e-11, -1.9200944777840952e-10, -1.8254439934330233e-15, 8.931042369641872e-16, 1.8035613221887612e-05, -1.8613231105256666e-06, -1.5526494905757578e-08, 9.427780669146273e-10]


# population[1]=[0.0, -1.1067969158138082e-12, -1.8830262421605557e-13, 5.183890790462502e-11, -2.1018074372779226e-10, -1.8569460954147734e-15, 8.770006038300017e-16, 1.9104900383948824e-05, -1.8944949659611762e-06, -1.5526494905757578e-08, 9.529917942449142e-10]

# population[2] =[0.0, -1.28344644319375e-12, -2.9253245131170985e-13, 4.9622242875292585e-11, -1.8589110909068555e-10, -1.814086038854568e-15, 7.357248758111538e-16, 3.066553210178919e-05, -1.865980228514031e-06, -1.4013925956299999e-08, 7.97454075648798e-10]


# population[3] =[0.0, -9.527204939247474e-13, -2.5125227001712807e-13, 5.2191973490828947e-11, -1.8869314670711164e-10, -1.799458100748353e-15, 7.727953885661946e-16, 2.8384758103728364e-05, -1.8436089754126478e-06, -1.2993197634180952e-08, 7.818948646520985e-10]


# population[4]= [0.0, -1.0977512913832238e-12, -1.7114270268631633e-13, 5.563642349484675e-11, -2.2338188123703347e-10, -1.6664413732220535e-15, 6.822128635972878e-16, 3.066553210178919e-05, -1.684144582766115e-06, -1.4013925956299999e-08, 7.410370972489066e-10]

# population[5] =[0.0, -1.1569281330280622e-12, -2.1529506619359898e-13, 5.656536695014186e-11, -1.9103603117376486e-10, -1.606426244590961e-15, 8.228360423748442e-16, 2.578642306470088e-05, -1.950536999377271e-06, -1.2121723447457567e-08, 8.218207234822871e-10]


# population[6] =[0.0, -1.1356883796076679e-12, -1.910739185423105e-13, 5.533008209108767e-11, -2.0462958543011666e-10, -1.7142178803107409e-15, 6.667219013595432e-16, 3.15306056556963e-05, -1.865980228514031e-06, -1.521524761515314e-08, 8.430104311599302e-10]


# population[7]=[0.0, -1.2776435968410938e-12, -2.8232841351888796e-13, 4.9622242875292585e-11, -1.8481055646937313e-10, -1.781439204191582e-15, 8.50455904210679e-16, 1.8944063904265665e-05, -2.0815276299213216e-06, -1.4614536362325702e-08, 9.776485050446484e-10]


# population[8] =[0.0, -1.1412550671737325e-12, -2.3219592968416814e-13, 5.3334943471277186e-11, -1.7212655663423219e-10, -1.936933832380068e-15, 6.046100535938544e-16, 2.9066863594267098e-05, -1.8360345151008853e-06, -1.3018083919393995e-08, 7.696962264424633e-10]


# population[9]=[0.0, -1.3827485652065593e-12, -3.221999622692096e-13, 5.5293829430497294e-11, -1.8079174147620345e-10, -2.2531981692202184e-15, 1.1041094669263413e-15, 2.8873465830249943e-05, -2.1346233025069927e-06, -1.1390167425186116e-08, 8.945449890728701e-10]



def fitness():
    print("Train and Validation error ")
    for i in range(10):
        train, validation = server.get_errors(
            'QXoVw3Dy9NvP3wdrvWwcPgFBYMSDDJTxsL1npdylcxC0j90Ff3', population[i])
        fit = (validation + (train)*train_factor)
        popfit.append(fit)

        print(train, validation)



def normFit():
    ss = np.sum(popfit)
    for i in range(10):
        popfit[i] = popfit[i]/ss
        popfit[i] = 1/popfit[i]

def fitness2():
    print("Train and Validation error ")
    for i in range(10):
        train, validation = server.get_errors(
            'QXoVw3Dy9NvP3wdrvWwcPgFBYMSDDJTxsL1npdylcxC0j90Ff3', offsprings[i])
        fit = (validation +  train*train_factor)
        popfit2.append(fit)

        print(train, validation)
    # print(popfit)


def normFit2():
    ss = np.sum(popfit2)
    for i in range(10):
        popfit2[i] = popfit2[i]/ss
        popfit2[i] = 1/popfit2[i]


def russianRoule():
    s = np.sum(popfit)
    print("SUM Norm")
    print(s)
    print()
    r = random.randint(0, int(s))
    p = r
    for v in range(10):
        p += popfit[v]
        if(p > s):
            return v
    return v-1


def crossover():
    mx = -1e17
    ti =0
    for i in range(len(popfit)):
            if(popfit[i] > mx):
                di = i
                mx = popfit[i]
    maxx = -1e17
    for i in range(len(popfit)):
            if(popfit[i] > maxx and i!=di):
                mi = i
                maxx = popfit[i]
    maxxx = -1e17
    for i in range(len(popfit)):
            if(popfit[i] > maxxx and i!=di and i!=mi):
                ti = i
                maxxx = popfit[i]
    print("After Selection (Parents Selected) ")    
    print(population[mi])
    print(population[di])
    print("----------------------------")
    for f in range(5):
        # di = russianRoule()
        # mi = russianRoule()
        # print("After Selection ")
        # print(mi)
        # print(di)
        # print()
        # fitness()
        # val = random.randint(1,3)
        # if val == 1 :
        mom = population[mi]
        dad = population[di]
        # elif val == 2:
        #     mom = population[di]
        #     dad = population[ti]
        # else :
        #     mom = population[ti]
        #     dad = population[mi]
        # part = np.random.randint(1, 9)
        part = 5
        # print(part)
        child = []
        child2 = []
        child = np.copy(dad)
        child2 = np.copy(mom)
        child2[0:part] = dad[0:part]
        child[0:part] = mom[0:part]
        c1 = child.tolist()
        c2 = child2.tolist()
        offsprings.append(c1)
        offsprings.append(c2)
        # for f in range(10):
        #     print(offsprings[f])
        # popfit = []


def mutate(vector):
    
    prob = 0.80
    for i in range(len(vector)):
        fact = random.uniform(-mutate_range, mutate_range)
        vector[i] = np.random.choice(
            [vector[i]*(fact+1), vector[i]], p=[prob, 1-prob])
        if(vector[i] < -10):
            vector[i] = -10
        elif(vector[i] > 10):
            vector[i] = 10

    return vector


# print(population[4])
for i in range(10):
        population[i] = mutate(population[i])


for x in range(5):
    for i in range(10):
        population[i] = mutate(population[i])
    print("Genetation iteration ",x)
    print("--- Initial Population ----")
    for pi in population:
        print(pi)
        print()
    print("----------------------------")
    fitness()
    normFit()
    crossover()
    print("---After Crossover----")
    for i in range(10):
        print(i,offsprings[i])
    print("----------------------------")
    #print offspring
    # popfit = []
    # population = []
    # population = offsprings
    for i in range(10):
        offsprings[i] = mutate(offsprings[i])
    print("----After Mutation--- ")
    for i in range(10):
        print(i,offsprings[i])
    print("----------------------------")
        #print 
    fitness2()
    normFit2()
    finallist = []
    for i in range(10):
        tup =  (population[i],popfit[i])
        finallist.append(tup)
        
    for i in range(10):
        tup = (offsprings[i],popfit2[i])
        finallist.append(tup)  


    finallist.sort(reverse=True , key=lambda x:x[1])

    for i in range(10):
        vec = finallist[i][0]
        population[i] = vec
    # print("FInallist")
    # for i in  range(20):

    #     print(finallist[i][0])
    #     print(finallist[i][1])
    print()
    offsprings = []
    
    print("-------------------NEW GENERATION-------------")
    for pi in population:
        print(pi)
        print()
    print("--------------------GENERATION ENDED-------------")
    print()
    print()
    popfit = []
    popfit2 = []
    offsprings = []
    # train_factor+=0.05

fitness()
normFit()
max = 0
for p in range(10):
    if (popfit[p] > max):
        max = p
server.submit(
    'QXoVw3Dy9NvP3wdrvWwcPgFBYMSDDJTxsL1npdylcxC0j90Ff3', population[p])
print("-------------------NEW GENERATION-------------")
print()
for pi in population:
    print(pi)
    print()
print("--------------------GENERATION ENDED-------------")
print()
print("Final Vector submitted")
print(population[p])
print()