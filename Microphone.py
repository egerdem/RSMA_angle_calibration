import numpy as np


class Microphone(object):

    def __init__(self, name='Generic', version='1.0', direct='Omnidirectional'):
        self._micname = name
        self._ver = version
        self._directivity = direct

    def getname(self):
        return self._micname

    def getversion(self):
        print(self._ver)

    def setname(self, name):
        self._micname = name

    def setversion(self, version):
        self._ver = version


class MicrophoneArray(Microphone):

    def __init__(self, name, typ, version, direct):
        super(MicrophoneArray, self).__init__(name, version, direct)
        self._arraytype = None
        self.__arraytype = typ

    def printtype(self):
        print(self.__arraytype)

    def settype(self, typ):
        self.__arraytype = typ


class EigenmikeEM32(MicrophoneArray):

    def __init__(self):
        super(EigenmikeEM32, self).__init__('Eigenmike 32', 'Rigid Spherical', 17.0, 'Omni')
        self._numelements = 32

        self._thetas = np.array([69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                               90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                               121.0, 159.0, 69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                               90.0, 125.0, 148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                               122.0, 159.0]) / 180.0 * np.pi

        self._phis = np.array([0.0, 32.0, 0.0, 328.0, 0.0, 45.0, 69.0, 45.0, 0.0, 315.0,
                             291.0, 315.0, 91.0, 90.0, 90.0, 89.0, 180.0, 212.0, 180.0, 148.0, 180.0,
                             225.0, 249.0, 225.0, 180.0, 135.0, 111.0, 135.0, 269.0, 270.0, 270.0,
                             271.0]) / 180.0 * np.pi

        self._radius = 4.2e-2

        self._weights = np.ones(32)

        # self._weights = np.array([1.0, 0.6539, 1.0, 0.6539, 0.6539, 1.0, 1.0, 1.0,
        #                         0.6539, 1.0, 1.0, 1.0, 1.0, 0.6539, 0.6539, 1.0, 1.0, 0.6539,
        #                         1.0, 0.6539, 0.6539, 1.0, 1.0, 1.0, 0.6539, 1.0, 1.0, 1.0,
        #                         1.0, 0.6539, 0.6539, 1.0])

        self._info = 'Eigenmike em32 needs to be calibrated using the software tool provided mh Acoustics before use.'
    def returnAsStruct(self):
        em32 = {'name': self._micname,
                'type': self._arraytype,
                'thetas': self._thetas,
                'phis': self._phis,
                'radius': self._radius,
                'weights': self._weights,
                'version': self._ver,
                'numelements': self._numelements,
                'directivity': self._directivity,
                'info': self._info}
        return em32


if __name__ == '__main__':
    a = EigenmikeEM32()
    em32 = a.returnAsStruct()
    print(em32)

