My first machine learning adventure courtesy [this tutorial](http://machinelearningmastery.com/machine-learning-in-python-step-by-step/).

### Setup:
I made a new virtualenv called scipy and `pip install scipy matplotlib pandas sklearn` which took a while. After I set up the file and tried to do my imports I ran into this error:

```
Traceback (most recent call last):
  File "ml.py", line 6, in <module>
    import matplotlib.pyplot as plt
  File "/Users/rdenton/.virtualenvs/scipy/lib/python2.7/site-packages/matplotlib/pyplot.py", line 114, in <module>
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
  File "/Users/rdenton/.virtualenvs/scipy/lib/python2.7/site-packages/matplotlib/backends/__init__.py", line 32, in pylab_setup
    globals(),locals(),[backend_name],0)
  File "/Users/rdenton/.virtualenvs/scipy/lib/python2.7/site-packages/matplotlib/backends/backend_macosx.py", line 24, in <module>
    from matplotlib.backends import _macosx
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are Working with Matplotlib in a virtual enviroment see 'Working with Matplotlib in Virtual environments' in the Matplotlib FAQ
```

To fix it I created a frameworkpython file in the bin directory of my scipy virtualenv like it talks about in the [FAQ](http://matplotlib.org/faq/virtualenv_faq.html#pythonhome-script). Then you run as `frameworkpython ml.py` instead of `python ml.py`.

I tried that but got a permission error so I changed the permissions from 413 to 755. After that everything worked fine. Well, there were a few warnings but nothing that broke anything. If you want to avoid the warnings you could install the same versions as the tutorial, some of mine were newer.

### Other issues

#### Figures
If you go about this the same way I do, you'll want to avoid `plt.show` because it opens up a window that you have to force quit, which forces a logout on your Terminal.

Instead do `plt.savefig('name.png')` and just save it as a .png in your directory.

#### Cross validation results
I got slightly different figures for the analysis and I'm guessing that's because you get random data so each result will be off but similar. Also, I'm using a different version so methods may have improved?





