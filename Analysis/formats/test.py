
def test():
    dual = []
    dual.append([0, lt[1]-Round])
    for i in range(1,len(lt)-1):
        dual.append([lt[i]+Round,  lt[i+1]-Round])
