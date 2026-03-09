# swiss_AI_discourse



## GPU Server usage

- upload (from local terminal)

```
scp -r <Key Directory> "<local folder path>" <user name>@<server name>.cl.uzh.ch:/home/<user name>/<folder_name>
```

- synchronize (from local terminal)

    - server to local
        ```
        rsync -asv --ignore-existing -e "<Key Directory>" <user name>@<server name>.cl.uzh.ch:/home/<user name>/<folder name> <local folder path>
        ```

    - local to server
        ```
        rsync -asv --ignore-existing -e "<Key Directory>" <local folder path> <user name>@<server name>.cl.uzh.ch:/home/<user name>/<folder name>
        ```

## Project Structure

- data: raw data and processed data (down load from query results)
    - file names: de.tsv, fr.tsv, de_edu.tsv,...

- topic_extraction: code for topic extraction
    