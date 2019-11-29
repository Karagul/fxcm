from collections import deque

m=deque(maxlen=10)

for i in range(100):

    k = m.append(i)
    print (k)