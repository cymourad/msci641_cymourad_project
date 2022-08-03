## Virtual Environment

We use virtual environment to manage dependencies.

To run it:

```powershell
.\msci641-env\Scripts\Activate.ps1
```

> I followed [this tutorial](https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/)

We also create a jupyter kernel to run form that virtual environment

```
ipython kernel install --user --name=msci641-env
```

We only ran that once, then we choose the kernel in VS Code or Jupyter notebook.

Once we are done the project, we can remove it form the list of jupyter kernels.

```
jupyter-kernelspec uninstall msci641-env
```

## Data

The data is too large to be uploaded onto GitHub. To get around that, I distilled the data that I actually need into `data/distil_data.csv`

# Tests

| BERT Layer | Overview Weight | Features Weight | Hits in 100        |
| ---------- | --------------- | --------------- | ------------------ |
| -2         | 1               | 1               | 0.6698442780162572 |
| -2         | 1               | 0               | 0.7132957347702986 |
| -2         | 0               | 1               | 0.9752032149054708 |
| -2         | 2               | 1               | 0.7068567905744817 |
| -2         | 3               | 1               | 0.7072449538770664 |
| -2         | 1               | 2               | 0.5599826468170609 |
| -1         | 1               | 0               | 0.7272810302310714 |
| -1         | 0               | 1               | 1.0020663987578775 |
| -1         | 1               | 2               | 0.4494931043930952 |
| -1         | 1               | 1               | 0.6733377477395196 |
| -1         | 2               | 1               | 0.7342679696775961 |
| -1         | 3               | 1               | 0.7184217736779615 |
