'''
Utility for logging experimental configuration.
'''

import codecs
from datetime import datetime
from collections import OrderedDict

def writeConfig(output, settings, title=None, start_time=None, end_time=None):
    '''Write an experimental configuration to a file.

    Always writes the current date and time at the head of the file.

    The optional title argument is a string to write at the head of the file,
        before date and time.

    Settings should be passed in as a list, in the desired order for writing.
    To write the value of a single setting, pass it as (name, value) pair.
    To group several settings under a section, pass a (name, dict) pair, where
        the first element is the name of the section, and the second is a
        dict (or OrderedDict) of { setting: value } format.

    For example, the following call:
        configlogger.writeConfig(some_path, [
            ('Value 1', 3),
            ('Some other setting', True),
            ('Section 1', OrderedDict([
                ('sub-value A', 12.4),
                ('sub-value B', 'string')
            ]))
        ], title='My experimental configuration')
    will produce the following configuration log:
        
        My experimental configuration
        Run time: 1970-01-01 00:00:00

        Value 1: 3
        Some other setting: True

        ## Section 1 ##
        sub-value A: 12.4
        sub-value B: string


    Arguments:

        output     :: str or stream; if str, uses as filepath to write to; if open stream,
                      writes to it but leaves stream open
        settings   :: (described above)
        title      :: optional string to write at start of config file
        start_time :: a datetime.datetime object indicating when the program started
                      execution; if not provided, defaults to datetime.now()
        end_time   :: a datetime.datetime object indicating when the program ended
                      execution; if provided, also writes elapsed execution time
                      between start_time and end_time
    '''

    group_set = set([dict, OrderedDict])

    # if passed in a filepath, open it here and close it when done
    if type(output) is str:
        output = codecs.open(output, 'w', 'utf-8')
        close_at_end = True
    else:
        close_at_end = False

    # headers
    if title:
        output.write('%s\n' % title)

    time_fmt = '%Y-%m-%d %H:%M:%S'

    if start_time is None:
        start_time = datetime.now()
        header = 'Run'
    else:
        header = 'Start'
    output.write('%s time: %s\n' % (header, start_time.strftime(time_fmt)))

    if end_time:
        output.write('End time: %s\n' % end_time.strftime(time_fmt))
        output.write('Execution time: %f seconds\n' % (end_time - start_time).total_seconds())
    output.write('\n')

    for (key, value) in settings:
        if type(value) in group_set:
            output.write('\n## %s ##\n' % key)
            for (sub_key, sub_value) in value.items():
                output.write('%s: %s\n' % (sub_key, str(sub_value)))
            output.write('\n')
        else:
            output.write('%s: %s\n' % (key, str(value)))

    if close_at_end:
        output.close()
