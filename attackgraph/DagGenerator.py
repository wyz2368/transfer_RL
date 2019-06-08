import networkx as nx
import numpy as np
import os
import random
from attackgraph import attacker
from attackgraph import defender
from attackgraph import file_op as fp
import copy

class Environment(object):
    #TODO: all representations are logically sorted.！！！！
    def __init__(self, num_attr_N = 11, num_attr_E = 4, T=10, numNodes=30, numEdges=100, numRoot=3, numGoals=6, history = 3):
        self.num_attr_N = num_attr_N
        self.num_attr_E = num_attr_E
        self.T = T
        self.current_time = 0
        self.history = history
        self.training_flag = -1 # =0 defender is training and =1 attacker is training and =2 both are training

        # randomDAG parameters
        self.numNodes = numNodes
        self.numEdges = numEdges
        self.numRoot = numRoot
        self.numGoals = numGoals

    #Initialize attacker and defender
    def create_players(self):
        #create players
        #TODO: test if the graph has been initialized.
        _, oredges = self.get_ORedges()
        _, andnodes = self.get_ANDnodes()
        self.attacker = attacker.Attacker(G=self.G,
                                          oredges=oredges,
                                          andnodes=andnodes,
                                          actionspace=self.get_att_actionspace())
        self.defender = defender.Defender(self.G, self.history)

    # Create action space for both players.
    def create_action_space(self):
        self.actionspace_att = self.get_att_actionspace()
        self.actionspace_def = self.get_def_actionspace()

    # Add attributes to current graph.
    def daggenerator_wo_attrs(self,nodeset,edgeset):
        # if not self.check_nodes_sorted(nodeset):
        nodeset = sorted(nodeset)
        edgeset = self.sortEdge(edgeset)

        self.G.add_nodes_from(nodeset,
                         root = 0, # 0:NONROOT 1:ROOTNODE
                         type = 0, # 0:NONTARGET 1:TARGET
                         eType = 0,# 0:OR 1:AND node
                         state = 0,# 0:Inactive 1:Active
                         aReward = 0.0, # GREATER THAN OR EQUAL TO 0
                         dPenalty = 0.0, # LESS THAN OR EQUAL TO 0
                         dCost = 0.0, # SAME^
                         aCost = 0.0, # SAME^
                         posActiveProb = 1.0, # prob of sending positive signal if node is active
                         posInactiveProb = 0.0, # prob of sending positive signal if node is inactive(false alarm)
                         actProb = 1.0) # prob of becoming active if being activated, for AND node only
        self.G.add_edges_from(edgeset,
                         eid = -1,
                         type = 0, # 0:NORMAL 1:VIRTUAL
                         cost = 0, # Cost for attacker on OR node, GREATER THAN OR EQUAL TO 0
                         actProb=1.0) # probability of successfully activating, for OR node only

    # Generate a random graph.
    def randomDAG(self, NmaxAReward=10, NmaxDPenalty=10, NmaxDCost=3, NmaxACost=3, EmaxACost=3):

        maxEdges = (self.numNodes-1)*(self.numNodes)/2
        if self.numEdges > maxEdges:
            raise Exception("For a graph with " + str(self.numNodes) + " nodes, there can be a maximum of " + str(int(maxEdges)) + " edges.")


        self.G = nx.gnp_random_graph(self.numNodes, 1, directed=True) # Create fully connected directed Erdos-Renyi graph.
        self.G = nx.DiGraph([(u,v) for (u,v) in self.G.edges() if u<v], horizon = self.T) # Drop all edges (u,v) where edge u<v to enforce acyclic graph property.
        rootNodes = random.sample(range(1,self.numNodes-1), self.numRoot-1) # Given the parameter self.numRoot, pick self.numRoot-1 random root IDs.
                                                                 # Node 0 will also always be root. Last node (ID:self.numNodes) cannot be root node.
        goalNodes = random.sample(list(set(range(1,self.numNodes))-set(rootNodes)),self.numGoals) # Randomly pick GoalNodes

        for rootNode in rootNodes: # Out of the picked rootNodes, drop all edges (u,v) where v = rootNode.
            for start in range(0, rootNode):
                self.G.remove_edge(start, rootNode)
        canRemove = list(self.G.edges)
        while len(self.G.edges) > self.numEdges and len(canRemove) != 0: # Randomly delete edges until self.numEdges is met, or if there are no more nodes to remove.
                                                                    # canRemove = nodes not yet removed OR nodes that once removed do not
                                                                    # break the connected property.
            deleteEdge = random.choice(canRemove)
            self.G.remove_edge(deleteEdge[0], deleteEdge[1])
            if (not nx.is_connected(self.G.to_undirected())) or (len(self.G.pred[deleteEdge[1]]) == 0):
                self.G.add_edge(deleteEdge[0], deleteEdge[1])
            canRemove.remove(deleteEdge)

        # Set random node attributes
        for nodeID in range(self.numNodes):
            if len(self.G.pred[nodeID]) == 0:
                self.setRoot_N(nodeID, 1)
                self.setType_N(nodeID, 0) # Root nodes cannot be target (goal) nodes.
                self.setActivationType_N(nodeID, 1) # Root nodes must be AND nodes
            else:
                self.setRoot_N(nodeID, 0)
                if nodeID in goalNodes: # Set Goal nodes
                    self.setType_N(nodeID, 1)
                else:
                    self.setType_N(nodeID, 0)
                self.setActivationType_N(nodeID, np.random.randint(2))

            self.setState_N(nodeID, 0)
            self.setAReward_N(nodeID, np.random.uniform(0, NmaxAReward))
            self.setDPenalty_N(nodeID, -np.random.uniform(0, NmaxDPenalty))
            self.setDCost_N(nodeID, -np.random.uniform(0, NmaxDCost))
            self.setACost_N(nodeID, -np.random.uniform(0, NmaxACost))
            self.setposActiveProb_N(nodeID, np.random.uniform(0.5, 1))
            self.setposInactiveProb_N(nodeID, np.random.uniform(0, 0.5))
            self.setActProb_N(nodeID, np.random.uniform(0.5, 1))

        # Nodes must start with id = 1
        self.G = nx.relabel_nodes(self.G, dict(zip(self.G.nodes, list(np.asarray(list(self.G.nodes)) + 1))))
        # This messes with the ordering of the edges; fix is below:
        sortedEdges = list(sorted(self.G.edges))
        for edge in sortedEdges:
            self.G.remove_edge(edge[0], edge[1])
        for edge in sortedEdges:
            self.G.add_edge(edge[0], edge[1])

        # Set random edge attributes
        for edgeID, edge in enumerate(self.G.edges):
            self.setid_E(edge, edgeID) # Edge ID could be random
            self.setType_E(edge, np.random.randint(2))
            self.setACost_E(edge, -np.random.uniform(0, EmaxACost))
            self.setActProb_E(edge, np.random.uniform(0.5, 1))

    # Assign new attributes to the graph.
    def specifiedDAG(self, attributesDict):
        for nodeID in attributesDict['nodes']:
            self.setRoot_N(nodeID, attributesDict['Nroots'][nodeID - 1])
            self.setType_N(nodeID, attributesDict['Ntypes'][nodeID - 1])
            self.setActivationType_N(nodeID, attributesDict['NeTypes'][nodeID - 1])
            self.setState_N(nodeID, attributesDict['Nstates'][nodeID - 1])
            self.setAReward_N(nodeID, attributesDict['NaRewards'][nodeID - 1])
            self.setDPenalty_N(nodeID, attributesDict['NdPenalties'][nodeID - 1])
            self.setDCost_N(nodeID, attributesDict['NdCosts'][nodeID - 1])
            self.setACost_N(nodeID, attributesDict['NaCosts'][nodeID - 1])
            self.setposActiveProb_N(nodeID, attributesDict['NposActiveProbs'][nodeID - 1])
            self.setposInactiveProb_N(nodeID, attributesDict['NposInactiveProbs'][nodeID - 1])
            self.setActProb_N(nodeID, attributesDict['NactProbs'][nodeID - 1])
        idx = 0
        for edge in attributesDict['edges']:
            self.setid_E(edge, attributesDict['Eeids'][idx])
            self.setType_E(edge, attributesDict['Etypes'][idx])
            self.setACost_E(edge, attributesDict['Ecosts'][idx])
            self.setActProb_E(edge, attributesDict['actProb'][idx])
            idx += 1

    def isProb(self,p):
        return p >= 0.0 and p <= 1.0

    def check_nodes_sorted(self,list):
        return all(list[i]<list[i+1] for i in range(len(list)-1))

    def sortEdge(self,edgeset):
        sorted_by_first_second = sorted(edgeset, key=lambda tup: (tup[0], tup[1]))
        return sorted_by_first_second

    # Graph Operation
    def getHorizon_G(self):
        return self.G.graph['horizon']

    def setHorizon_G(self,value):
        self.T = value
        self.G.graph['horizon'] = value

    # Node Operations
    # Get Info
    def isOrType_N(self,id):
        return self.G.nodes[id]['eType'] == 0

    def getState_N(self,id):
        return self.G.nodes[id]['state']

    def getType_N(self,id): # Target or NonTarget
        return self.G.nodes[id]['type']

    def getActivationType_N(self,id): # AND or OR
        return self.G.nodes[id]['eType']

    def getAReward_N(self,id):
        return self.G.nodes[id]['aReward']

    def getDPenalty_N(self,id):
        return self.G.nodes[id]['dPenalty']

    def getDCost_N(self,id):
        return self.G.nodes[id]['dCost']

    def getACost_N(self,id):
        return self.G.nodes[id]['aCost']

    def getActProb_N(self,id):
        return self.G.nodes[id]['actProb']

    def getposActiveProb_N(self,id):
        return self.G.nodes[id]['posActiveProb']

    def getposInactiveProb_N(self,id):
        return self.G.nodes[id]['posInactiveProb']

    # Set Info
    # id is the node number rather than the index.

    def setRoot_N(self, id, value):
        if value != 0 and value != 1:
            raise ValueError("Node root value must be 0 (NONROOT) or 1 (ROOT).")
        else:
            self.G.nodes[id]['root'] = value

    def setState_N(self,id,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node state value must be 0 (Inactive) or 1 (Active).")
            else:
                self.G.nodes[id]['state'] = value
        except Exception as error:
            print(repr(error))

    def setType_N(self,id,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node type must be 0 (NONTARGET) or 1 (TARGET).")
            else:
                self.G.nodes[id]['type'] = value
        except Exception as error:
            print(repr(error))

    def setActivationType_N(self,id,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Node eType must be 0 (OR) or 1 (AND).")
            else:
                self.G.nodes[id]['eType'] = value
        except Exception as error:
            print(repr(error))

    def setAReward_N(self,id,value):
        try:
            if value < 0:
                raise Exception("Node aReward must be greater than or equal to 0.")
            else:
                self.G.nodes[id]['aReward'] = value
        except Exception as error:
            print(repr(error))

    def setDPenalty_N(self,id,value):
        try:
            if value > 0:
                raise Exception("Node dPenalty must be less than or equal to 0.")
            else:
                self.G.nodes[id]['dPenalty'] = value
        except Exception as error:
            print(repr(error))

    def setDCost_N(self,id,value):
        try:
            if value > 0:
                raise Exception("Node dCost must be less than or equal to 0.")
            else:
                self.G.nodes[id]['dCost'] = value
        except Exception as error:
            print(repr(error))

    def setACost_N(self,id,value):
        try:
            if value > 0:
                raise Exception("Node aCost must be less than or equal to 0.")
            else:
                self.G.nodes[id]['aCost'] = value
        except Exception as error:
            print(repr(error))

    def setActProb_N(self,id,value):
        try:
            if not self.isProb(value):
                raise Exception("Node action probability is not a valid probability (0>=<=1).")
            else:
                self.G.nodes[id]['actProb'] = value
        except Exception as error:
            print(repr(error))

    def setposActiveProb_N(self,id,value):
        try:
            if not self.isProb(value):
                raise Exception("Node posActive probability is not a valid probability (0>=<=1).")
            else:
                self.G.nodes[id]['posActiveProb'] = value
        except Exception as error:
            print(repr(error))

    def setposInactiveProb_N(self,id,value):
        try:
            if not self.isProb(value):
                raise Exception("Node posInctive probability is not a valid probability (0>=<=1).")
            else:
                self.G.nodes[id]['posInactiveProb'] = value
        except Exception as error:
            print(repr(error))

    # Edge Operations
    # Get Info
    def getid_E(self,edge):
        return self.G.edges[edge]['eid']

    def getType_E(self,edge): # 0:NORMAL 1:VIRTUAL
        return self.G.edges[edge]['type']

    def getACost_E(self,edge):
        return self.G.edges[edge]['cost']

    def getActProb_E(self,edge):
        return self.G.edges[edge]['actProb']


    # Set Info

    def setid_E(self, edge, value):
        self.G.edges[edge]['eid'] = value

    def setType_E(self,edge,value):
        try:
            if value != 0 and value != 1:
                raise Exception("Edge type must be either 0 (Normal) or 1 (Virtual).")
            else:
                 self.G.edges[edge]['type'] = value
        except Exception as error:
            print(repr(error))

    def setACost_E(self,edge,value):
        try:
            if value > 0:
                raise Exception("Cost for attacker on edge to OR node must be greater than or equal to 0.")
            else:
                 self.G.edges[edge]['cost'] = value
        except Exception as error:
            print(repr(error))

    def setActProb_E(self,edge,value):
        try:
            if not self.isProb(value):
                raise Exception("Probability of successfully activating node through edge (For OR nodes only) is not a valid probability (0>=<=1).")
            else:
                self.G.edges[edge]['actProb'] = value
        except Exception as error:
            print(repr(error))

    # Print Info
    def print_N(self,id):
        print(self.G.nodes[id])

    def print_E(self,edge):
        print(self.G.edges[edge])

    # Other Operations
    def getNumNodes(self):
        return self.G.number_of_nodes()

    def getNumEdges(self):
        return self.G.number_of_edges()

    def inDegree(self,id):
        return self.G.in_degree(id)

    def outDegree(self,id):
        return self.G.out_degree(id)

    def predecessors(self,id):
        return set(self.G.predecessors(id))

    def successors(self,id):
        return set(self.G.successors(id))

    def isDAG(self):
        return nx.is_directed_acyclic_graph(self.G)

    def getEdges(self):
        return self.G.edges()

    def get_ANDnodes(self):
        count = 0
        Andset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['eType'] == 1:
                count += 1
                Andset.add(node)
        return count, Andset

    def get_ORnodes(self):
        count = 0
        Orset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['eType'] == 0:
                count += 1
                Orset.add(node)
        return count, Orset

    def get_ORedges(self):
        _, ornodes = self.get_ORnodes()
        oredges = []
        for node in ornodes:
            oredges += self.G.in_edges(node)
        oredges = self.sortEdge(oredges)
        return len(oredges), oredges

    def get_Targets(self):
        count = 0
        targetset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['type'] == 1:
                count += 1
                targetset.add(node)
        return count,targetset

    def get_Roots(self):
        count = 0
        rootset = set()
        for node in self.G.nodes:
            if self.G.nodes[node]['root'] == 1:
                count += 1
                rootset.add(node)
        return count,rootset

    def get_NormalEdges(self):
        count = 0
        normaledge = set()
        for edge in self.G.edges:
            if self.G.edges[edge]['type'] == 0:
                count += 1
                normaledge.add(edge)
        return count, self.sortEdge(normaledge)

    # Attributes Initialization
    def assignAttr_N(self,id,attr): #add code to check the lenth match
        self.G.nodes[id].update(dict(zip(self.G.nodes[id].keys(),attr)))

    def assignAttr_E(self,edge,attr):
        self.G.edges[edge].update(dict(zip(self.G.edges[edge].keys(),attr)))

    ### API FUNCTIONS ###

    #return a list of indicator of whether node is activated.
    def get_att_isActive(self):
        isActive = []
        for id in self.G.nodes:
            if self.G.nodes[id]['state'] == 1:
                isActive.append(1)
            else:
                isActive.append(0)
        return isActive

    # can be called only once for each env time step since the randomness.
    def get_def_hadAlert(self):
        alert = []
        for node in self.G.nodes:
            if self.G.nodes[node]['state'] == 1:
                if random.uniform(0, 1) <= self.G.nodes[node]['posActiveProb']:
                    alert.append(1)
                else:
                    alert.append(0)
            elif self.G.nodes[node]['state'] == 0:
                if random.uniform(0, 1) <= self.G.nodes[node]['posInactiveProb']:
                    alert.append(1)
                else:
                    alert.append(0)
            else:
                raise ValueError("node state is abnormal.")
        return alert

    def get_att_actionspace(self):
        _, andnodes = self.get_ANDnodes()
        _, oregdes = self.get_ORedges()
        actionspace = sorted(list(andnodes)) + oregdes + ['pass']
        return actionspace

    def get_def_actionspace(self):
        actionspace = list(self.G.nodes) + ['pass']
        return actionspace

    # step function while action set building

    # attact and defact are attack set and defence set#
    def _step(self, done = False):
        # immediate reward for both players
        aReward = 0
        dReward = 0
        # Strategy should be assigned to players for every episode.
        if self.training_flag == 0: # If the defender is training, attacker builds greedy set. Vice Versa.
            self.attacker.att_greedy_action_builder(self.G, self.T - self.current_time + 1)
        elif self.training_flag == 1:
            self.defender.def_greedy_action_builder(self.G, self.T - self.current_time + 1)
        else:
            raise ValueError("training flag error.")

        attact = self.attacker.attact #TODO: check if action sets are correct!
        defact = self.defender.defact
        # attacker's action
        for attack in attact:
            if isinstance(attack, tuple):
                # check OR node
                aReward += self.G.edges[attack]['cost']
                if random.uniform(0, 1) <= self.G.edges[attack]['actProb']:
                    self.G.nodes[attack[-1]]['state'] = 1
            else:
                # check AND node
                aReward += self.G.nodes[attack]['aCost']
                if random.uniform(0, 1) <= self.G.nodes[attack]['actProb']:
                    self.G.nodes[attack]['state'] = 1
        # defender's action
        for node in defact:
            self.G.nodes[node]['state'] = 0
            dReward += self.G.nodes[node]['dCost']
        _, targetset = self.get_Targets()
        for node in targetset:
            if self.G.nodes[node]['state'] == 1:
                aReward += self.G.nodes[node]['aReward']
                dReward += self.G.nodes[node]['dPenalty']

        if self.training_flag == 0: # defender is training
            # attacker updates obs
            self.attacker.update_obs(self.get_att_isActive())
            # defender updates obs and defact
            self.defender.update_obs(self.get_def_hadAlert())
            self.defender.save_defact2prev()
            self.defender.defact.clear()
            inDefenseSet = self.defender.get_def_inDefenseSet(self.G) #should be all zeros.
            wasdef = self.defender.get_def_wasDefended(self.G)
            return np.array(self.defender.prev_obs + self.defender.observation + \
               wasdef + inDefenseSet + [self.T - self.current_time]), dReward, done

        elif self.training_flag == 1: # attacker is training
            # defender updates obs and defact. Don't need to clear defset since it's done in greedy builder.
            self.defender.update_obs(self.get_def_hadAlert())
            self.defender.save_defact2prev()
            self.defender.defact.clear()
            # attacker updates obs
            self.attacker.update_obs(self.get_att_isActive())
            self.attacker.attact.clear()
            canAttack, inAttackSet = self.attacker.get_att_canAttack_inAttackSet(self.G)
            self.attacker.update_canAttack(canAttack)
            # inAttackSet should be all zeros, we can check this.
            return np.array(self.attacker.observation + canAttack + inAttackSet + [self.T - self.current_time]), aReward, done
        else:
            raise ValueError("Training flag is set abnormally.")

    # action is a number index for the real action。
    # These two functions are for inner greedy action set building.
    def _step_att(self, action):
        immediatereward = 0
        self.attacker.attact.add(action)
        canAttack, inAttackset = self.attacker.get_att_canAttack_inAttackSet(self.G)
        # Notice that canAttack is updated in the _step.
        return np.array(self.attacker.observation + canAttack + inAttackset + [self.T - self.current_time]), \
               immediatereward, False

    def _step_def(self, action):
        immediatereward = 0
        self.defender.defact.add(action)
        inDefenseSet = self.defender.get_def_inDefenseSet(self.G)
        wasdef = self.defender.get_def_wasDefended(self.G) # May be improved since it's the same within internal clock.
        return np.array(self.defender.prev_obs + self.defender.observation + \
               wasdef + inDefenseSet + [self.T - self.current_time]), immediatereward, False


    # Environment step function
    def step(self, action):
        if self.training_flag == 0: #defender is training.
            true_action = self.actionspace_def[action]
            if true_action == 'pass' or true_action in self.defender.defact:
                self.current_time += 1
                if self.current_time < self.T:
                    new_obs, rew, done = self._step()
                    return new_obs, rew, done
                else:
                    new_obs, rew, done = self._step(done=True)
                    return new_obs, rew, done
            else:
                new_obs, rew, done = self._step_def(true_action)
                return new_obs, rew, done
        elif self.training_flag == 1: # attacker is training.
            true_action = self.actionspace_att[action]
            if true_action == 'pass' or true_action in self.attacker.attact:
                self.current_time += 1
                if self.current_time < self.T:
                    new_obs, rew, done = self._step()
                    return new_obs, rew, done
                else:
                    new_obs, rew, done = self._step(done=True)
                    return new_obs, rew, done
            else:
                new_obs, rew, done = self._step_att(true_action)
                return new_obs, rew, done
        else:
            raise ValueError("In step function, training_flag is invalid.")


    #reset the environment, G_reserved is a copy of the initial env
    def save_graph_copy(self):
        self.G_reserved = copy.deepcopy(self.G)

    def save_mask_copy(self):
        self.G_mask = copy.deepcopy(self.G)

    def reset_graph(self):
        self.G = copy.deepcopy(self.G_reserved)

    def reset_everything_with_return(self):
        #TODO: Does not finish.
        self.current_time = 0
        self.reset_graph()
        self.attacker.reset_att()
        self.defender.reset_def()
        if self.training_flag == 0: # defender is training.
            self.defender.observation = [0]*self.G.number_of_nodes()
            inDefenseSet = [0]*self.G.number_of_nodes()
            wasdef = [0]*self.G.number_of_nodes()*self.history #TODO: here is history, not history - 1
            return np.array(self.defender.prev_obs + self.defender.observation + wasdef + inDefenseSet + [self.T - self.current_time])
        elif self.training_flag == 1: # attacker is training.
            self.attacker.update_obs([0]*self.G.number_of_nodes())
            canAttack, inAttackset = self.attacker.get_att_canAttack_inAttackSet(self.G)
            #canAttack should be root nodes.
            self.attacker.update_canAttack(canAttack)
            return np.array(self.attacker.observation + canAttack + inAttackset + [self.T - self.current_time]) # t0=0
        else:
            raise ValueError("Training flag is abnormal.")

    def reset_everything(self):
        #TODO: Does not finish.
        self.current_time = 0
        self.reset_graph()
        self.attacker.reset_att()
        self.defender.reset_def()
        self.training_flag = -1


    #other APIs similar to OpenAI gym
    def obs_dim_att(self): # Change this after removing "canAttack".
        num_andnode, _ = self.get_ANDnodes()
        num_oredges, _ = self.get_ORedges()
        return self.G.number_of_nodes() + 2*(num_andnode + num_oredges) + 1

    def obs_dim_def(self):
        N = self.G.number_of_nodes()
        return self.history*N*2 + N + 1 # NO CNN

    def act_dim_att(self):
        num_andnode, _ = self.get_ANDnodes()
        num_oredges, _ = self.get_ORedges()
        return num_andnode + num_oredges + 1 #pass

    def act_dim_def(self):
        return self.G.number_of_nodes() + 1 #pass

    def set_training_flag(self, flag):
        self.training_flag = flag

    def set_current_time(self,time):
        self.current_time = time

def env_rand_gen_and_save(env_name, num_attr_N = 11, num_attr_E = 4, T=10, numNodes=30, numEdges=100, numRoot=4, numGoals=6, history = 3):
    env = Environment(num_attr_N = num_attr_N, num_attr_E = num_attr_E, T=T, numNodes=numNodes, numEdges=numEdges, numRoot=numRoot, numGoals=numGoals, history = history)
    env.randomDAG()
    path = os.getcwd() + "/env_data/" + env_name + ".pkl"
    print("env path is ", path)
    # if fp.isExist(path):
    #     raise ValueError("Env with such name already exists.")
    fp.save_pkl(env,path)
    print(env_name + " has been saved.")
    return env


