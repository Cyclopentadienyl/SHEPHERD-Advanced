"""
SHEPHERD-Advanced Data Sources Module
=====================================
資料來源整合模組

支援的資料來源:
- Ontologies: HPO, MONDO, GO, MP
- Gene-Disease DBs: DisGeNET, ClinVar, OMIM, Orphanet
- Literature: PubMed, Pubtator
- Cross-Species: MGI, ZFIN, Ensembl, OrthoDB, PANTHER
- Drug: DrugBank, ChEMBL

使用方式:
    from src.data_sources import PubMedDataSource, OrthologDataSource

    # 文獻資料
    pubmed = PubMedDataSource(mode='offline')
    pubmed.load_pubtator_db('/path/to/pubtator.db')

    # 同源基因
    ortholog = OrthologDataSource()
    mappings = ortholog.get_orthologs('BRCA1', species=Species.MOUSE)
"""

from src.core.protocols import (
    DataSourceProtocol,
    PubMedDataSourceProtocol,
    PubtatorLocalDBProtocol,
    OrthologDataSourceProtocol,
)
from src.core.types import DataSource, Species

__all__ = [
    # Protocols (for type hints)
    "DataSourceProtocol",
    "PubMedDataSourceProtocol",
    "PubtatorLocalDBProtocol",
    "OrthologDataSourceProtocol",
    # Enums
    "DataSource",
    "Species",
]
