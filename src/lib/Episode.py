from __future__ import absolute_import, print_function

from collections.abc import Sequence

from src.lib.Sample import Sample, toStr


class Episode(Sequence):
    """ Episode is a collection of samples """
    def __init__(self, samples):
        for sample in samples:
            assert(isinstance(sample, Sample))

        self.samples = samples
        super(Episode, self).__init__()

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return ' >,< '.join([toStr(sample) for sample in self.samples])
