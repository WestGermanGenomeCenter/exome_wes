for SRR in \
SRR36189783 \
SRR36189784 \
SRR36189785 \
SRR36189787 \
SRR36189922 \
SRR36189924 \
SRR36189926 \
SRR36189928
do
    curl -s \
    "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${SRR}&result=read_run&fields=fastq_ftp" \
    | tail -n1 \
    | cut -f2 \
    | tr ';' '\n'
done | sed 's#^#https://#' > urls.txt

wget -c -i urls.txt


# the samples in the top dont seem to work well, so downloading new data:
#!/usr/bin/env bash

for SRR in \
SRR36189783 \
SRR36189784 \
SRR36189785 \
SRR36189787 \
SRR36189922 \
SRR36189924 \
SRR36189926 \
SRR36189928
do
    curl -s \
    "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${SRR}&result=read_run&fields=fastq_ftp" |
    tail -n1 |
    cut -f2 |
    tr ';' '\n' |
    while read -r ftp_path
    do
        file=$(basename "$ftp_path")
        echo "curl -L https://$ftp_path -o $file"
    done
done > download.sh

chmod +x download.sh