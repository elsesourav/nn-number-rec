// get index form array 
function getIndexWhereOne(array) {
   let max = {
      i: 0,
      val: 0
   }
   for (let i = 0; i < array.length; i++) {
      if (max.val < array[i]) {
         max.i = i;
         max.val = array[i];
      }
   }
   return max.i;
}