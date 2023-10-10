This repository is an attempt to make the panphon version (current best model for multilanguage) to reach 100% in homophone generation.

My initial guess was that the panphon representation contained a lot of repetitions. This guess proved false.

I tried to add token digits in the input together with vector representation, but the actual behavior diveged from the desired in the sense that the output was biased towards the digits not considering sound.

To make thhe model faster I also reduced the number of allowed IPA tokens, which considerably degraded the multilanguage performance.

Batch reduction proved effective for the models without panphon representation, but not for this model.

https://gist.githubusercontent.com/tomByrer/cb5c9fae362c896ecd02/raw/f2fd9eae8b65c8a68bd3b80f3c4bd3115497da90/english-homophones.txt
http://www.singularis.ltd.uk/bifroest/misc/homophones-list.html