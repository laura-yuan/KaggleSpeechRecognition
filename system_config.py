def get_system_config():
    import csv
    # first, get into the folder of system config....
    system_config = {}
    with open('systemConfig.csv') as f:
        content = csv.reader(f, delimiter = ';')
        for row in content:
            system_config[row[0]] = row[1]
    return system_config