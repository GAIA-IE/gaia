# -*- coding: utf-8 -*-

import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.setrecursionlimit(100000)
sys.path.append(os.path.abspath(""))
os.chdir(os.path.join(os.path.abspath(""), "pSCRDRtagger"))

from multiprocessing import Pool
from InitialTagger.InitialTagger import initializeSentence
from SCRDRlearner.Object import FWObject
from SCRDRlearner.SCRDRTree import SCRDRTree
from Utility.Config import NUMBER_OF_PROCESSES
from Utility.Utils import getWordTag, readDictionary


def unwrap_self_RDRPOSTagger(arg, **kwarg):
    return RDRPOSTagger.tagRawSentence(*arg, **kwarg)


class RDRPOSTagger(SCRDRTree):
    """
    RDRPOSTagger for a particular language
    """

    def __init__(self):
        self.root = None

    def tagRawSentence(self, DICT, rawLine):
        line = initializeSentence(DICT, rawLine)
        sen = []
        wordTags = line.split()
        tag_sequence = []
        for i in xrange(len(wordTags)):
            fwObject = FWObject.getFWObject(wordTags, i)
            word, tag = getWordTag(wordTags[i])
            node = self.findFiredNode(fwObject)
            if node.depth > 0:
                sen.append(word + "||" + node.conclusion)
            else:  # Fired at root, return initialized tag
                sen.append(word + "||" + tag)
        return " ".join(sen)

    def tagRawCorpus(self, DICT, lines):
        # Change the value of NUMBER_OF_PROCESSES to obtain faster tagging process!
        pool = Pool(processes=NUMBER_OF_PROCESSES)
        taggedLines = pool.map(unwrap_self_RDRPOSTagger, zip([self] * len(lines), [DICT] * len(lines), lines))
        print "\n".join(taggedLines)


def printHelp():
    print "\n===== Usage ====="
    print '\n#1: To train RDRPOSTagger on a gold standard training corpus:'
    print '\npython RDRPOSTagger.py train PATH-TO-GOLD-STANDARD-TRAINING-CORPUS'
    print '\nExample: python RDRPOSTagger.py train ../data/goldTrain'
    print '\n#2: To use the trained model for POS tagging on a raw text corpus:'
    print '\npython RDRPOSTagger.py tag PATH-TO-TRAINED-MODEL PATH-TO-LEXICON PATH-TO-RAW-TEXT-CORPUS'
    print '\nExample: python RDRPOSTagger.py tag ../data/goldTrain.RDR ../data/goldTrain.DICT ../data/rawTest'
    print '\n#3: Find the full usage at http://rdrpostagger.sourceforge.net !'


def run(args=sys.argv[1:]):
    if (len(args) == 0):
        printHelp()
    elif args[0].lower() == "tag":
        try:
            r = RDRPOSTagger()
            r.constructSCRDRtreeFromRDRfile(args[1])
            DICT = readDictionary(args[2])
            r.tagRawCorpus(DICT, sys.stdin.readlines())
        except Exception, e:
            print "\nERROR ==> ", e
            printHelp()
    else:
        printHelp()


if __name__ == "__main__":
    run()
    pass
