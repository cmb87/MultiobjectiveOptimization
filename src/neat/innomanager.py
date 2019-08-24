




class InnovationManager:
    def __init__(self):
        self.innovations = {}
        self.seenInnovations = {}
        self.innovationCtr = 0

    ### Add innovation ###
    def add(self, n1, n2, generation=None):
        ### I have already seen this one ###
        if '{}-{}'.format(n1, n2) in list(self.seenInnovations.keys()):
            return self.seenInnovations['{}-{}'.format(n1, n2)]

        ### This combination doesn't exist yet, adding it ... ###
        self.innovationCtr+=1
        self.seenInnovations['{}-{}'.format(n1, n2)] = self.innovationCtr
        self.innovations[self.innovationCtr] = {"nodes": [n1, n2], "generation": generation}
        print('New innovation ({}) {}-{} added!'.format(self.innovationCtr, n1, n2))
        return self.innovationCtr


    ### return nids and connection ###
    def getFeatureFromIno(self, inos):

        nids = list(set([n for n in self.innovations[ino]["nodes"] for ino in inos]))
