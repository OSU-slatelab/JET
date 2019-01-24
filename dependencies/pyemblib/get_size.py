'''
Print the dimensionality of an embedding file to stdout
'''
import sys
from . import *

if __name__=='__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMBF')
        parser.add_option('-d', '--dim-only', dest='dim_only',
            action='store_true', default=False,
            help='Only print out dimensionality')
        (options, args) = parser.parse_args()
        if len(args) != 1:
            parser.print_help()
            exit()
        return args[0], options.dim_only
    embf, dim_only = _cli()
    (nwords, dim) = getSize(embf)
    if dim_only:
        sys.stdout.write('%d\n' % dim)
    else:
        sys.stdout.write('%d %d\n' % (nwords, dim))
