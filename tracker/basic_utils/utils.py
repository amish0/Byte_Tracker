import os
from pathlib import Path

def check_file(file, suffix=''):
    """
    @brief check file is valid or not
    @param file file path
    @param suffix file suffix
    @return file path if file is valid else empty string
    """
    if check_suffix(file, suffix):  # optional
        file = str(file)  # convert to str()
        if os.path.isfile(file) or not file:  # exists
            return file
        else:
            print('File Not Found: %s' % file)
            return ''
    else:
        print('File suffix doesnot match: %s' % file)
        return ''
    
def check_suffix(file: str='sort.yaml', suffix: str='.yaml', msg=''):
    """
    @brief check file suffix
    @param file file path
    @param suffix file suffix
    @param msg message to be printed
    @return True if suffix matches else False
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        else:
            print('suffix must be string')
            return False
        # check file is string or windowsPath
        if isinstance(file, Path):
            file = str(file)
        if isinstance(file, str):
            s = Path(file).suffix.lower()  # file suffix
            if s in suffix:
                return True
            else:
                print('suffix: {}'.format(s))
                print('suffix acceptable: {}'.format(suffix))
                return False
        else:
            print('file must be string')
            return False
    else:
        print('file and suffix must be string')
        return False