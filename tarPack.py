import tarfile


with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('export', recursive=True)
