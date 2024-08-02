The examples rely on data in a separate ragpipe repository. 

```
git clone https://github.com/ragpipe/data/ my/data/folder
```

Point the `etc.data_folder` variable in `config.yml` to `my/data/folder`. Paths are relative to the main ragpipe folder.

## Insurance 

```
examples/insurance/
|
|-- insurance.py
|-- insurance.yml
```

```bash 
python -m examples.insurance.insurance
```

## Startups

```bash 
python -m examples.startups
```


## Billionaires

```bash 
python -m examples.billionaires
```