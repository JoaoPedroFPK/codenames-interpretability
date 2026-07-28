"""Remote execution transport for GPU stages run on Colab.

This subpackage carries **no methodology**. It moves ``codenames-experiment``
invocations to a GPU session and moves their status, logs, and artifacts back.
The frozen contract, the seed, and every numeric path stay in the modules that
already own them.
"""
