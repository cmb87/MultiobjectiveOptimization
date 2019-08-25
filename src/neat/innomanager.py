

class InnovationManager:
    def __init__(self):
        self.innovationArchive = {}
        self.innovationCtr = 0
        self.nodectr = 0

    ### Add innovation ###
    def add(self, n1, n2, level, generation=None):
        ### name of the invention ###
        name = '{}-{}-{}'.format(level, n1, n2)
        ### Get highest node id ###
        self.nodectr = max([n1 ,n2, self.nodectr])

        ### I have already seen this one ###
        if name in list(self.innovationArchive.keys()):
            return self.innovationArchive[name]["innovationNumber"]

        ### This combination doesn't exist yet, adding it ... ###
        self.innovationCtr+=1
        self.innovationArchive[name] = {"nodes": [n1, n2], "generation": generation, "level": level,
                                        "innovationNumber": self.innovationCtr}

        print('New innovation ({}) {} added!'.format(self.innovationCtr, name))
        return self.innovationCtr

    ### for mutate add node ###
    def getNewNodeID(self):
        self.nodectr+=1
        print("New node ({}) added!".format(self.nodectr))
        return self.nodectr

    ### Return nodes ###
    def getNodes(self, ino):
        return [val for key, val in self.innovationArchive.items() if val["innovationNumber"] == ino][0]




        
