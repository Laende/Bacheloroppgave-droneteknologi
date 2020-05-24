from filecmp import dircmp

dir1 = '<DIR 1>'
dir2 = '<DIR 2'


def print_diff_files(dcmp):
    for name in dcmp.diff_files:
        print(f"diff_file {name}")
        for sub_dcmp in dcmp.subdirs.values():
            print_diff_files(sub_dcmp)

# Sammenligner filene i to forskjellige mapper
dcmp = dircmp(dir1, dir2)

# Sende forskjellene til funksjonen definert over
print_diff_files(dcmp)