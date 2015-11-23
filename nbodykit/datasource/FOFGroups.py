from nbodykit.plugins import DataSource
import numpy
import logging
from nbodykit.utils import selectionlanguage

logger = logging.getLogger('FOFGroups')

def list_str(value):
    return value.split()
         
class FOFDataSource(DataSource):
    """
    Class to read field data from a HDF5 FOFGroup data file

    Notes
    -----
    * `h5py` must be installed to use this data source.
    
    Parameters
    ----------
    path    : str
        the path of the file to read the data from 
    dataset: list of str
        For text files, one or more strings specifying the names of the data
        columns. Shape must be equal to number of columns
        in the field, otherwise, behavior is undefined.
        For hdf5 files, the name of the pandas data group.
    BoxSize : float or array_like (3,)
        the box size, either provided as a single float (isotropic)
        or an array of the sizes of the three dimensions
    usecols : list of str, optional
         if not None, only these columns will be read from file
    poscols : list of str, optional
        Full path to the column of the position vector
    velcols : list of str, optional
        Full path to the column of the velocity vector
    masscols : list of str, optional
        Full path to the column of the mass
    rsd     : [x|y|z], optional
        direction to do the redshift space distortion
    posf    : float, optional
        multiply the position data by this factor
    velf    : float, optional
        multiply the velocity data by this factor
    select  : str, optional
        string specifying how to select a subset of data, based
        on the column names. For example, if there are columns
        `type` and `mass`, you could specify 
        select= "type == central and mass > 1e14"
    """
    field_type = "FOFGroups"
    
    @classmethod
    def register(kls):
        
        h = kls.add_parser()
        
        h.add_argument("path", help="path to file")
        h.add_argument("BoxSize", type=kls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions.")
        h.add_argument("m0", type=float, help="mass unit")
        h.add_argument("-dataset",  default="FOFGroups", help="name of dataset in HDF5 file")
        h.add_argument("-rsd", choices="xyz", 
            help="direction to do redshift distortion")
        h.add_argument("-select", default=None, type=selectionlanguage.Query, 
            help='row selection based on conditions specified as string')
    
    def readall(self, columns):
        try:
            import h5py
        except:
            raise ImportError("h5py must be installed to use HDF5 reader")
                
        dataset = h5py.File(self.path, mode='r')[self.dataset]
        data = dataset[...]

        data2 = numpy.empty(len(data),
            dtype=[
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3)),
                ('Mass', 'f4'),
                ('Weight', 'f4'),
                ('Length', 'i4'),
                ('Rank', 'i4'),
                ('LogMass', 'f4')])

        data2['Mass'] = data['Length'] * self.m0
        data2['Weight'] = 1.0
        data2['LogMass'] = numpy.log10(data2['Mass'])
        # get position and velocity, if we have it
        data2['Position'] = data['Position'] * self.BoxSize
        data2['Velocity'] = data['Velocity'] * self.BoxSize
        data2['Rank'] = numpy.arange(len(data))
        # select based on input conditions
        if self.select is not None:
            mask = self.select.get_mask(data2)
            data2 = data2[mask]

        nobj = (len(data2), len(data))

        logger.info("total number of objects selected is %d / %d" % nobj)

        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            data2['Position'][:, dir] += data2['Velocity'][:, dir]
            data2['Position'][:, dir] %= self.BoxSize[dir]

        return [data2[key].copy() for key in columns]
