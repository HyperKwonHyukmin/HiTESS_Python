 private void ModifyUboltPos()
 {
     List<AMPipe> uboltList = amModel.PipeModels.FindAll(s => s.Type == "UBOLT");
     List<AMPipe> tubiList = amModel.PipeModels.FindAll(s => s.Type == "TUBI");
     AMPipe nearTubi;
     List<AMStru> nearStruList;
     Point3D tempUboltPos = new Point3D();
     Point3D tempStruPos = new Point3D();
     List<Point3D> tempPosList;
     //List<string> nameList = new List<string>();
     foreach (AMPipe ubolt in uboltList)
     {

         // ubolt 위치가 tubi상에 있는지 확인하여 near Tubi 찾음.
         nearTubi = tubiList.Find(s => s.Bran == ubolt.Bran && ModelHandle.IsPointOnLineBetweenTwoPoints(ubolt.Pos, s.APos, s.LPos) && ModelHandle.GetDistanceClosestPoint(ubolt.Pos, s.APos, s.LPos) <= 20);
         // ubolt와 stru Volume으로 근처 stru를 찾음.
         
         nearStruList = GetNearStruToUbolt(ubolt);
         
         // 250626 psh pipe가 큰 경우 panel을 연장하는데 그러면 가까운 support를 찾을 수 없음. 관경이 큰 경우, 범위를 넓혀서 다시 찾기
         if (nearTubi == null)
             continue;
         if (nearStruList.Count == 0 && nearTubi.OutDia > 650)
         {
             nearStruList = amModel.StruModels.FindAll(s => isNearStru(ubolt, s, 20));
         }

         //nameList.Add(GetNearStruToUbolt1(ubolt));
         if (nearTubi == null || nearStruList.Count == 0)
             continue;
         foreach (AMStru nearStru in nearStruList)
         {
             tempUboltPos = ModelHandle.GetClosestPoint(nearTubi, nearStru);
             if (tempUboltPos == null || tempUboltPos == new Point3D(0, 0, 0))
                 continue;
             else
                 break;

         }
         if (tempUboltPos == null || tempUboltPos == new Point3D(0, 0, 0))
             continue;
         else
             ubolt.Pos = tempUboltPos;
         if (ubolt.Remark == "BOX")
         {
             foreach (AMStru stru in nearStruList)
             {
                 //tempStruPos = ModelHandle.GetClosestPoint(stru, nearTubi);
                 //if (tempStruPos == null || tempStruPos == new Point3D(0, 0, 0))
                 //    continue;
                 // 원래 위치에서  Ubolt 범위에 있는지 체크
                 if (ModelHandle.IsParallel(stru, nearTubi))
                     continue;
                 tempStruPos = ModelHandle.GetNearestPointOnA( stru.OriPoss,stru.OriPose, nearTubi,false);
                 if (isInUbolt(stru.Rad, ubolt.Wvol, ModelHandle.GetDistance(ubolt.Pos, tempStruPos)) && ModelHandle.IsPointOnLineBetweenTwoPoints(tempStruPos, stru.OriPoss, stru.OriPose))
                 {
                     // 반영하는 위치는 변경된 위치로
                     tempStruPos = ModelHandle.GetNearestPointOnA(stru, nearTubi);
                     tempPosList = ubolt.InterPos.ToList();
                     tempPosList.Add(tempStruPos);
                     ubolt.InterPos = tempPosList;
                     tempPosList = stru.InterPos.ToList();
                     tempPosList.Add(tempStruPos);
                     stru.InterPos = tempPosList;
                 }
             }
         }
         else
         {
             double minDist = 10000;

             Point3D minDistStruPos = new Point3D(0, 0, 0);
             int minIdx = 100;
             for (int i = 0; i < nearStruList.Count; i++)
             {
                 tempStruPos = ModelHandle.GetClosestPoint(nearStruList[i], nearTubi);
                 if (tempStruPos == null || tempStruPos == new Point3D(0, 0, 0))
                     continue;
                 if (ModelHandle.GetDistance(tempStruPos, tempUboltPos) < minDist)
                 {
                     minDistStruPos = tempStruPos;
                     minDist = ModelHandle.GetDistance(tempStruPos, tempUboltPos);
                     minIdx = i;
                 }
             }
             if (minDist == 1000 || minIdx == 100)
                 continue;
             tempPosList = ubolt.InterPos.ToList();
             tempPosList.Add(minDistStruPos);
             ubolt.InterPos = tempPosList;
             tempPosList = nearStruList[minIdx].InterPos.ToList();
             tempPosList.Add(minDistStruPos);
             nearStruList[minIdx].InterPos = tempPosList;
             // check 이동된 좌표에서 tubi와 stru가 서로 벗어나 있을수 있음.
             // 이 경우, tubi의 끝단중 가까운곳에 연결
             if (!ModelHandle.IsPointOnLineBetweenTwoPoints(tempUboltPos, nearTubi.APos, nearTubi.LPos))
             {
                 if (ModelHandle.GetDistance(tempUboltPos, nearTubi.APos) <= ModelHandle.GetDistance(tempUboltPos, nearTubi.LPos))
                     ubolt.Pos = nearTubi.APos;
                 else
                     ubolt.Pos = nearTubi.LPos;
             }


         }

     }
     //nameList = nameList.Distinct().ToList();
 }
