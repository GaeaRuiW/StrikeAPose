import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from tracker import matching
from .basetrack import BaseTrack, TrackState

# 修改1：在STrack类中添加cls_id和keypoints属性
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls_id=None, keypoints=None):  # 添加额外参数
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.cls_id = cls_id  # 新增：类别ID
        self.keypoints = keypoints  # 新增：关键点数据
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        # 修改2：更新类别ID和关键点数据
        self.cls_id = new_track.cls_id
        self.keypoints = new_track.keypoints

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        # 修改3：更新类别ID和关键点数据
        self.cls_id = new_track.cls_id
        self.keypoints = new_track.keypoints

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

# 修改4：在BYTETracker类中添加处理额外信息的功能
class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results):
        # 修改5：接受额外信息（cls_ids和keypoints）
        if isinstance(output_results, tuple) and len(output_results) == 3:
            # 新格式：(detections, cls_ids, keypoints)
            detections, cls_ids, keypoints = output_results
        else:
            # 原始格式
            detections = output_results
            cls_ids = None
            keypoints = None
        
        self.frame_id += 1
        activated_starcks = [] #保存当前帧匹配到持续追踪的轨迹
        refind_stracks = [] #保存当前帧匹配到之前目标丢失的轨迹
        lost_stracks = [] #保存当前帧没有匹配到目标的轨迹
        removed_stracks = [] #保存当前帧
        # 第一步：将objects转换为x1，y1，x2，y2，score的格式，并构建strack
        if detections.shape[1] == 5:
            scores = detections[:, 4]
            bboxes = detections[:, :4]
        else:
            detections = detections.cpu().numpy()
            scores = detections[:, 4] * detections[:, 5]
            bboxes = detections[:, :4]  # x1y1x2y2

        #第二步：根据scroe和track_thresh将strack分为detetions(dets)(>=)和detections_low(dets_second)
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)   #  筛选分数处于0.1<分数<阈值的
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        
        # 修改6：处理额外的类别ID和关键点
        if cls_ids is not None:
            cls_ids_keep = cls_ids[remain_inds] if len(cls_ids) > 0 else None
            cls_ids_second = cls_ids[inds_second] if len(cls_ids) > 0 else None
        else:
            cls_ids_keep = None
            cls_ids_second = None
            
        if keypoints is not None:
            keypoints_keep = keypoints[remain_inds] if len(keypoints) > 0 else None
            keypoints_second = keypoints[inds_second] if len(keypoints) > 0 else None
        else:
            keypoints_keep = None
            keypoints_second = None

        if len(dets) > 0:
            '''Detections'''
            # 修改7：在创建STrack时传递额外信息
            detections = []
            for i, (tlbr, s) in enumerate(zip(dets, scores_keep)):
                cls_id = cls_ids_keep[i] if cls_ids_keep is not None else None
                kps = keypoints_keep[i] if keypoints_keep is not None else None
                detections.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, cls_id, kps))
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        # 遍历tracked_stracks（所有的轨迹），如果track还activated，加入tracked_stracks（继续匹配该帧），否则加入unconfirmed
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        # is_activated表示除了第一帧外中途只出现一次的目标轨迹（新轨迹，没有匹配过或从未匹配到其他轨迹）
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 第一次匹配
        # 将track_stracks和lost_stracks合并得到track_pool
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # 将strack_pool送入muti_predict进行预测（卡尔曼滤波）
        STrack.multi_predict(strack_pool)
        # 计算strack_pool（当前帧的预测框和之前未匹配到轨迹的bbox）和detections的iou_distance(代价矩阵)
        # detections是当前帧的bbox
        # 如果矩阵没有交集则为1
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 用match_thresh = 0.8(越大说明iou越小)过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
        # matches = [itracked, idet] itracked指的是轨迹的索引，idet 指的是当前目标框的索引，意思是第几个轨迹匹配第几个目标框
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 更新轨迹的bbox为当前匹配到的bbox
                track.update(detections[idet], self.frame_id)
                # activated_starcks是目前能持续追踪到的轨迹
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                # refind_stracks是重新追踪到的轨迹
                refind_stracks.append(track)
        # 第二次匹配：和低分的矩阵进行匹配
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            # 修改8：在创建STrack时传递额外信息
            detections_second = []
            for i, (tlbr, s) in enumerate(zip(dets_second, scores_second)):
                cls_id = cls_ids_second[i] if cls_ids_second is not None else None
                kps = keypoints_second[i] if keypoints_second is not None else None
                detections_second.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, cls_id, kps))
        else:
            detections_second = []
        # 找出第一次匹配中没匹配到的轨迹（激活状态）
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算r_tracked_stracks和detections_second的iou_distance(代价矩阵)
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 用match_thresh = 0.8过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5) #分数比较低的目标框没有匹配到轨迹就会直接被扔掉，不会创建新的轨迹
        # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # 遍历u_track（第二次匹配也没匹配到的轨迹），将state不是Lost的轨迹，调用mark_losk方法，并加入lost_stracks，等待下一帧匹配
        # lost_stracks加入上一帧还在持续追踪但是这一帧两次匹配不到的轨迹
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, which tracks with only one beginning frame'''
        # 尝试匹配中途第一次出现的轨迹
        # 当前帧的目标框会优先和长期存在的轨迹（包括持续追踪的和断追的轨迹）匹配，再和只出现过一次的目标框匹配
        detections = [detections[i] for i in u_detection]
        # 计算unconfirmed和detections的iou_distance(代价矩阵)
        # unconfirmed是不活跃的轨迹（过了30帧）
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 用match_thresh = 0.8过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        # 遍历u_unconfirmed，调用mark_removd方法，并加入removed_stracks
        for it in u_unconfirmed:
            # 中途出现一次的轨迹和当前目标框匹配失败，删除该轨迹（认为是检测器误判）
            # 真的需要直接删除吗？？
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 遍历u_detection（前两步都没匹配成功的目标框），对于score大于high_thresh，调用activate方法，并加入activated_stracks
        # 此时还没匹配的u_detection将赋予新的id
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # 只有第一帧新建的轨迹会被标记为is_activate=True，其他帧不会
            track.activate(self.kalman_filter, self.frame_id)
            #把新的轨迹加入到当前活跃轨迹中
            activated_starcks.append(track)
        # 遍历lost_stracks，对于丢失超过max_time_lost(30)的轨迹，调用mark_removed方法，并加入removed_stracks
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        # 遍历tracked_stracks，筛选出state为Tracked的轨迹，保存到tracked_stracks
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 将activated_stracks，refind_stracks合并到track_stracks
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 遍历lost_stracks,去除tracked_stracks和removed_stracks中存在的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # 调用remove_duplicate_stracks函数，计算tracked_stracks，lost_stracks的iou_distance，对于iou_distance<0.15的认为是同一个轨迹，
        # 对比该轨迹在track_stracks和lost_stracks的跟踪帧数和长短，仅保留长的那个
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        # 遍历tracked_stracks，将所有的is_activated为true的轨迹输出
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb