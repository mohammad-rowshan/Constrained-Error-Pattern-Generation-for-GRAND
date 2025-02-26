# Constrained Error Pattern Generation for ORB-GRAND
If you find this algorithm useful, please cite the following paper. Thanks.

M. Rowshan and J. Yuan, "Constrained Error Pattern Generation for GRAND," 2022 IEEE International Symposium on Information Theory (ISIT), Espoo, Finland, 2022, pp. 1767-1772, doi: 10.1109/ISIT50566.2022.9834343.

[https://ieeexplore.ieee.org/document/9354542](https://ieeexplore.ieee.org/abstract/document/9834343)

Abstract: Maximum-likelihood (ML) decoding can be used to obtain the optimal performance of error correction codes. However, the size of the search space and consequently the decoding complexity grows exponentially, making it impractical to be employed for long codes. In this paper, we propose an approach to constrain the search space for error patterns under a recently introduced near ML decoding scheme called guessing random additive noise decoding (GRAND). In this approach, the syndrome-based constraints which divide the search space into disjoint sets are progressively evaluated. By employing $p$ constraints extracted from the parity check matrix, the average number of queries reduces by a factor of $2^p$ while the error correction performance remains intact.

The work was improved by Segmentation in the following paper:

M. Rowshan and J. Yuan, "Low-Complexity GRAND by Segmentation," GLOBECOM 2023 - 2023 IEEE Global Communications Conference, Kuala Lumpur, Malaysia, 2023, pp. 6145-6151, doi: 10.1109/GLOBECOM54140.2023.10436895.

[https://ieeexplore.ieee.org/abstract/document/9328621](https://ieeexplore.ieee.org/abstract/document/10436895/)

For the script, please see the repository https://github.com/mohammad-rowshan/Segmented-GRAND

Please report any bugs to mrowshan at ieee dot org
