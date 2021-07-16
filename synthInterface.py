# Generic sound class to store sound synthesizer files
import numpy as np
import math

class MyParam():
    def __init__(self,name,min,max, val, cb,synth_doc) :
        self.name=name
        self.min=min
        self.max=max
        self.val=val
        self.cb=cb
        self.synth_doc = synth_doc

    # only store the actual value, not the normed value used for setting
    def __setParamNorm__(self, i_val) :
        self.val=self.min + i_val * (self.max - self.min);

##################################################################################################
#  the base sound model from which all synths should derive
##################################################################################################
'''
    A model has parameters, methods to get/set them, and a generate function that returns a signal.
    This is *the* interface for all synths.
'''
class MySoundModel() :

    def __init__(self,sr=16000, rng=None) :
        self.param = {} # a dictionary of MyParams
        self.sr = sr
        if rng==None :
            print(f"{self.__class__.__name__} creating new RNG")
            self.rng = np.random.default_rng(18005551212)
        else :
            self.rng = rng


    def __addParam__(self, name,min,max,val, cb=None, synth_doc="") :
        self.param[name]=MyParam(name,min,max,val, cb,synth_doc)


    def setParam(self, name, value) :
        if self.param[name].cb is not None :
            self.param[name].cb(value)
        self.param[name].val=value

    ''' set parameters using [0,1] which gets mapped to [min, max] '''
    def setParamNorm(self, name, nvalue) :
        if self.param[name].cb is not None :
            self.param[name].cb(self.getParam(name))
        self.param[name].__setParamNorm__(nvalue)

    def getParam(self, name, prop="val") :
        if prop == "val" :
            return self.param[name].val
        if prop == "min" :
            return self.param[name].min
        if prop == "max" :
            return self.param[name].max
        if prop == "name" :
            return self.param[name].name
        if prop == "synth_doc" :
            return self.param[name].synth_doc

    ''' returns list of paramter names that can be set by the user '''
    def getParams(self) :
        plist=[]
        for p in self.param :
            plist.append(self.param[p].name)
        return plist

    '''
        override this for your signal generation
    '''
    def generate(self, sigLenSecs=1) :
        return np.zeros(sigLenSecs*self.sr)

    ''' returns list of paramter names and their ranges '''
    def paramProps(self) :
        plist=[]
        for p in self.param :
            plist.append(self.param[p])
        return plist

    ''' Print all the parameters and their ranges from the synth'''
    def printParams(self):
        paramVals = self.paramProps()
        for params in paramVals:
            print( "Name: ", params.name, " Current value : ", params.val, " Max value ", params.max, " Min value ", params.min, "Synth Doc", params.synth_doc )

##################################################################################################
# A Ensemble class for playing a bunch of DSSynth models together
##################################################################################################
'''
    A DSSynth for creating a bunch of models that play at the same time.
    Create each in the usual way, setting their parameters, etc. Then pass them as an array to DSEnsemble.
    The factory takes an optional argument for a list of amplitudes that, if used, must be the same length as the models list.
    The generate function takes a spreadSecs argument that lest you spread out start times evenly over an interval.
'''
class DSEnsemble(MySoundModel) :
    def __init__(self,  models=[], amp=[], rng=None) :
        MySoundModel.__init__(self)
        self.numModels= len(models)
        self.models=models
        if len(amp) != len(models) :
            print(f'will use uniform amplitudes unless len(amps) == len(models)')
            amp=np.ones(len(models))*.6
        self.amp=amp

    def generate(self,  durationSecs, spreadSecs=1) :
        numSamples=int(self.sr*durationSecs)
        spreadsamples=int(self.sr*spreadSecs)

        sig=np.zeros(numSamples+spreadsamples)
        for i in range (self.numModels) :
            gensig = self.amp[i]*self.models[i].generate(durationSecs)
            sig = addin(gensig, sig, self.rng.integers(0,spreadsamples))
        return sig[:numSamples]


##################################################################################################
# A couple of handy-dandy UTILITY FUNCTIONS for event pattern synthesizers in particular
##################################################################################################
'''
creates a list of event times that happen with a rate of 2^r_exp
      and deviate from the strict equal space according to irreg_exp

      @rng - If None, uses fixed seed for repeatability. Otherwise, provide your own, seeded for repeatability or not.
                default: rng = np.random.default_rng(18005551212)
      @wrap - mode by duration so that anything that fell off either end is wrapped back in to [0,durationSecs]
      @roll - shift all events so that first one starts at time 0
'''
def noisySpacingTimeList(rate_exp, irreg_exp, durationSecs,  rng=None, verbose=False, wrap=True, roll=False) :

    if rng==None :
        rng = np.random.default_rng(18005551212)

    # mapping to the right range units
    eps=np.power(2.,rate_exp)
    irregularity=.1*irreg_exp*np.power(10,irreg_exp)
    #irregularity=.04*np.power(10,irreg_exp)
    sd=irregularity/eps

    #print("eps = {}, irreg = {}, and sd = {}".format(eps, irregularity, sd))

    linspacesteps=int(eps*durationSecs)
    linspacedur = linspacesteps/eps

    if verbose :
        print(f'noisySpacingTimeList: eps is {eps}, sd = {sd}, linspacesteps is {linspacesteps}, linspacedur is {linspacedur}')

    eventtimes=[(x+rng.normal(scale=sd))%durationSecs for x in np.linspace(0, linspacedur, linspacesteps, endpoint=False)]

    if verbose :
        print(f'noisySpacingTimeList: (BEFORE wrapped, rolled) eventtimes =  {eventtimes}')


    if wrap :
        eventtimes=np.sort(np.mod(eventtimes, durationSecs))
    if roll :
        eventtimes=eventtimes-np.min(eventtimes)

    if verbose :
        print(f'noisySpacingTimeList: (wrapped, rolled) eventtimes =  {eventtimes}')

    return eventtimes #sort because we "wrap around" any events that go off the edge of [0. durationSecs]



''' convert a list of floats (time in siconds) to a signal with pulses at those time '''
def timeList2Sig(elist, sr, durationSecs) :
    numsamps=sr*durationSecs
    sig=np.zeros(numsamps)
    for nf in elist :
        sampnum=int(round(nf*sr))
        if sampnum<numsamps and sampnum >= 0 :
            sig[sampnum]=1
        else :
            print("in timeList2Sig, warning: sampnum(={}) out of range".format(sampnum))
    return sig



'''adds one (shorter) array (a) in to another (b) starting at startsamp in b'''
def addin(a,b,startsamp) :
    b[startsamp:startsamp+len(a)]=[sum(x) for x in zip(b[startsamp:startsamp+len(a)], a)]
    return b

''' Returns a chunked wav files from generated signal '''
def selectVariation(sig, sr, varNum, varDurationSecs):
        variationSamples=math.floor(sr*varDurationSecs)
        return sig[varNum*variationSamples:(varNum+1)*variationSamples]


'''Gestures are transformation function specifying changes about an aspect of a
sound over time. Used for creating amplitude envelopes or frequency sweeps.'''

'''Linearly interpolates from start to stop val
   Startval: Float, int
   Stopval: Float, int
'''
def gesture(startVal, stopVal, cutOff, numSamples):
        gesture = np.zeros(numSamples)
        non_zero = np.linspace(startVal, stopVal, int(cutOff*numSamples))
        for index in range(len(non_zero)):
                gesture[index] = non_zero[index]
        return gesture

'''Generic gesture creates 2 linear interpolations.'''
''' Startval: Float, int
    Stopval: Float, int
    2 interpolations: Start to stop, and stop to start
'''
def genericGesture(startVal, stopVal, cutOff, numSamples):
        gesture = np.zeros(numSamples)
        ascending = np.linspace(startVal, stopVal, int(cutOff*numSamples))
        descending = np.linspace(stopVal, startVal, numSamples - int(cutOff*numSamples))

        for index in range(len(ascending)):
            gesture[index] = ascending[index]
        for index in range(len(descending)):
            gesture[index+len(ascending)] = descending[index]

        return gesture

''' Create an array comprised of linear segments between breakpoints '''
# y - list of values
# s - list of number of samples to interpolate between sucessive values
def bkpoint(y,s) :
    assert(len(y)==(len(s)+1))
    sig=[]
    for j in range(len(y)-1) :
        sig=np.concatenate((sig, np.linspace(y[j], y[j+1], s[j], False)), 0)
    return sig

def oct2freq(octs, bf=440.) :
    return bf * np.power(2,octs)

def freq2oct(freq, bf=440.) :
    return np.log2(freq/bf)
