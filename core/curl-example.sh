curl -X 'PUT' \
  'http://127.0.0.1:8000/api/v1/actions/update_action' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "action_id": 1,
  "data": {
    "step_length": [
      {
        "step_id": 0,
        "size": 113.31,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 1,
        "size": 230.15,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 2,
        "size": 44.00,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 3,
        "size": 189.05,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 4,
        "size": 158.32,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 5,
        "size": 197.29,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 6,
        "size": 182.12,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 7,
        "size": 114.50,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 8,
        "size": 309.28,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 9,
        "size": 126.41,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 10,
        "size": 324.19,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 11,
        "size": 17.99,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 12,
        "size": 368.93,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 13,
        "size": 4.56,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 14,
        "size": 462.81,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 15,
        "size": 98.00,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 16,
        "size": 384.43,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 17,
        "size": 169.64,
        "left_feet": false,
        "right_feet": true
      }
    ],
    "step_speed": [
      {
        "step_id": 0,
        "size": 1699.6918010711681,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 1,
        "size": 373.2098904171505,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 2,
        "size": 125.7204328264509,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 3,
        "size": 872.5298353341909,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 4,
        "size": 339.2575740814219,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 5,
        "size": 696.3185658174402,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 6,
        "size": 376.80514171205715,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 7,
        "size": 361.56563206722814,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 8,
        "size": 639.8996234762261,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 9,
        "size": 399.2043023360389,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 10,
        "size": 648.3892466227217,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 11,
        "size": 67.46431350708065,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 12,
        "size": 737.8544413248692,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 13,
        "size": 13.667131423948376,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 14,
        "size": 991.7355918884274,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 15,
        "size": 326.6550487942156,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 16,
        "size": 1002.873256517494,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 17,
        "size": 565.4759597778336,
        "left_feet": false,
        "right_feet": true
      }
    ],
    "step_stride": [
      {
        "stride_id": 0,
        "size": 343.45888582865405
      },
      {
        "stride_id": 1,
        "size": 233.05028247833252
      },
      {
        "stride_id": 2,
        "size": 355.6104615529383
      },
      {
        "stride_id": 3,
        "size": 296.6182686487832
      },
      {
        "stride_id": 4,
        "size": 435.6995137532549
      },
      {
        "stride_id": 5,
        "size": 342.185106913249
      },
      {
        "stride_id": 6,
        "size": 373.48293113708405
      },
      {
        "stride_id": 7,
        "size": 560.8064575195308
      },
      {
        "stride_id": 8,
        "size": 554.0775362650561
      }
    ],
    "step_hip_degree": [
      {
        "step_id": 0,
        "low": 3.4063195652285034,
        "high": 4.501510660244816,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 1,
        "low": 0.04465765230494525,
        "high": 12.115537958683204,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 2,
        "low": 0.1404305458749648,
        "high": 9.73677661802761,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 3,
        "low": 0.2824748594069699,
        "high": 8.956758504933594,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 4,
        "low": 2.0712179252804215,
        "high": 23.864039575642913,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 5,
        "low": 1.565798641821248,
        "high": 20.40171895902014,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 6,
        "low": 0.7105509149512657,
        "high": 22.173505109751417,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 7,
        "low": 0.3772682728257605,
        "high": 20.728664940498774,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 8,
        "low": 0.02920273931178381,
        "high": 20.483370857041063,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 9,
        "low": 1.9350313508552228,
        "high": 10.091057573409392,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 10,
        "low": 0.3817000137416123,
        "high": 6.065970147646394,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 11,
        "low": 0.0033530235462126496,
        "high": 11.183781173306757,
        "left_feet": false,
        "right_feet": true
      },
      {
        "step_id": 12,
        "low": 0.3678589704716584,
        "high": 12.780035842510843,
        "left_feet": true,
        "right_feet": false
      },
      {
        "step_id": 13,
        "low": 1.2159930197798638,
        "high": 7.697541598309249,
        "left_feet": false,
        "right_feet": true
      }
    ]
  }
}
'