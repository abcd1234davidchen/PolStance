manually dropped:
【更多內容 請見影片】訂閱【自 由追新聞】全新的視界！新聞話題不漏接，快訂閱YouTube 【自由追新聞】，記得開啟小鈴鐺哦！
訂閱【自由追新聞】全新的視界！新聞 話題不漏接，快訂閱YouTube 【自由追新聞】，記得開啟小鈴鐺哦！
// 創物件 var tvPlayer = new VideoAPI_LiTV(); // 設定自動播放 tvPlayer.setAutoplay(true); //不自動播放 tvPlayer.setDelay(0); // 設定延遲 tvPlayer.setAllowFullscreen(true); tvPlayer.setType('web'); // tvPlayer.setControls(1); litv 無法操作顯示控制項 tvPlayer.pushVideoIdByClassName('TVPlayer', tvPlayer); setTimeout(function (){ tvPlayer.loadAPIScript('cache_video_js_LiTV'); },3000)