#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#include <math.h>

#import <objc/runtime.h>
#import <CoreVideo/CoreVideo.h>
// Private Espresso buffer ABI on this macOS build; only the data pointer is consumed.
typedef struct {void *data;uint64_t opaque[19];int tail;} EBuffer;
@interface NSObject (DirectEspresso)
- (id)espressoResources;
- (BOOL)bindBuffer:(EBuffer *)b toNetworkOutputBlobName:(NSString *)n error:(NSError **)e;
- (BOOL)bindPixelBuffer:(CVPixelBufferRef)b toNetworkBlobName:(NSString *)n error:(NSError **)e;
- (BOOL)executePlanAndReturnError:(NSError **)e;
@end
static NSString *captureRoot;
static NSUInteger captureIndex;
static NSMutableArray *captureRecords;
typedef id (*ProcessFn)(id,SEL,CGRect,CVPixelBufferRef,id,unsigned,id,NSError **,id);
static ProcessFn originalProcess;
static id captureProcess(id self,SEL selector,CGRect roi,CVPixelBufferRef pixel,id opts,unsigned qos,id recorder,NSError **err,id progress) {
 if(pixel) {
  CVPixelBufferLockBaseAddress(pixel,kCVPixelBufferLock_ReadOnly);
  size_t width=CVPixelBufferGetWidth(pixel),height=CVPixelBufferGetHeight(pixel),stride=CVPixelBufferGetBytesPerRow(pixel);
  OSType format=CVPixelBufferGetPixelFormatType(pixel);
  NSString *name=[NSString stringWithFormat:@"%04lu-%lu.raw",(unsigned long)captureIndex,(unsigned long)captureRecords.count];
  NSData *data=[NSData dataWithBytes:CVPixelBufferGetBaseAddress(pixel) length:height*stride];
  if(![data writeToFile:[captureRoot stringByAppendingPathComponent:name] atomically:YES])abort();
  [captureRecords addObject:@{@"file":name,@"width":@(width),@"height":@(height),@"stride":@(stride),@"format":@(format)}];
  CVPixelBufferUnlockBaseAddress(pixel,kCVPixelBufferLock_ReadOnly);
 }
 id result=originalProcess(self,selector,roi,pixel,opts,qos,recorder,err,progress);
 for(NSString *name in @[@"flatten_output",@"confidence"]) {
  EBuffer b={0};NSError *error=nil;
  BOOL ok=[self bindBuffer:&b toNetworkOutputBlobName:name error:&error];
  if(!ok||!b.data){NSLog(@"BIND %@ %@",name,error);continue;}
  size_t n=[name isEqualToString:@"flatten_output"]?128:[name isEqualToString:@"confidence"]?1:3*128*128;
  NSString *filename=[NSString stringWithFormat:@"%04lu-%@.raw",(unsigned long)captureIndex,name];
  [[NSData dataWithBytes:b.data length:n*4] writeToFile:[captureRoot stringByAppendingPathComponent:filename] atomically:YES];
 }
 return result;
}
static void installCapture(void) {
 Class c=NSClassFromString(@"VNFaceAnalyzerMultiDetectorANFRv4_MD2");
 SEL sel=NSSelectorFromString(@"processRegionOfInterest:croppedPixelBuffer:options:qosClass:warningRecorder:error:progressHandler:");
 Method method=class_getInstanceMethod(c,sel);
 if(!method){fprintf(stderr,"capture method unavailable\\n");abort();}
 originalProcess=(ProcessFn)method_getImplementation(method);
 if(!class_addMethod(c,sel,(IMP)captureProcess,method_getTypeEncoding(method)))
  class_replaceMethod(c,sel,(IMP)captureProcess,method_getTypeEncoding(method));
}


// Private API, inspected on macOS 15.8.2. Runtime checks fail closed on API changes.
@interface NSObject (PrivateFaceprint)
- (id)faceprint;
- (CGRect)unalignedBoundingBox;
- (NSData *)descriptorData;
- (NSUInteger)elementCount;
- (NSUInteger)elementType;
- (float)confidence;
- (id)originatingRequestSpecifier;
- (void)setForceFaceprintCreation:(BOOL)value;
- (void)setInputFaceObservations:(NSArray *)observations;
@end

static NSDictionary *score(NSDictionary *item) {
    NSMutableDictionary *row=[item mutableCopy];
    captureRecords=[NSMutableArray new]; captureIndex++;
    NSError *error=nil;
    VNImageRequestHandler *handler=[[VNImageRequestHandler alloc]
        initWithURL:[NSURL fileURLWithPath:item[@"path"]] options:@{}];
    VNDetectFaceRectanglesRequest *detect=[VNDetectFaceRectanglesRequest new];
    if(![handler performRequests:@[detect] error:&error]) {
        row[@"error"]=error.localizedDescription; return row;
    }
    row[@"face_count"]=@(detect.results.count);
    if(!detect.results.count) { row[@"error"]=@"no_face_detected"; return row; }
    NSArray *faces=[detect.results sortedArrayUsingComparator:^NSComparisonResult(VNFaceObservation *a,VNFaceObservation *b) {
        double x=a.boundingBox.size.width*a.boundingBox.size.height;
        double y=b.boundingBox.size.width*b.boundingBox.size.height;
        return x>y?NSOrderedAscending:x<y?NSOrderedDescending:NSOrderedSame;
    }];
    VNFaceObservation *face=faces.firstObject;
    CGRect b=face.boundingBox;
    CGRect rawBox=[face unalignedBoundingBox];
    row[@"detector_box"]=@[@(rawBox.origin.x),@(rawBox.origin.y),@(rawBox.size.width),@(rawBox.size.height)];
    row[@"bbox"]=@[@(b.origin.x),@(b.origin.y),@(b.size.width),@(b.size.height)];
    row[@"detection_confidence"]=@(face.confidence);
    row[@"second_face_area_ratio"]=faces.count>1?
        @(((VNFaceObservation *)faces[1]).boundingBox.size.width*((VNFaceObservation *)faces[1]).boundingBox.size.height/(b.size.width*b.size.height)):@0;
    Class cls=NSClassFromString(@"VNCreateFaceprintRequest");
    VNImageBasedRequest *print=[[cls alloc] init];
    if(!print || ![print respondsToSelector:@selector(setForceFaceprintCreation:)]) {
        row[@"error"]=@"private_faceprint_API_unavailable"; return row;
    }
    print.usesCPUOnly=YES;
    [print setForceFaceprintCreation:YES];
    [print setInputFaceObservations:@[face]];
    row[@"faceprint_request_revision"]=@(print.revision);
    if(![handler performRequests:@[print] error:&error]) {
        row[@"error"]=error.localizedDescription; return row;
    }
    id fp=[print.results.firstObject faceprint];
    NSData *data=[fp descriptorData];
    if(!fp || [fp elementType]!=1 || [fp elementCount]!=128 || data.length!=128*sizeof(float)) {
        row[@"error"]=@"unexpected_faceprint_shape_or_type"; return row;
    }
    const float *values=data.bytes;
    NSMutableArray *embedding=[NSMutableArray new];
    double norm2=0;
    for(int i=0;i<128;i++) {
        if(!isfinite(values[i])) { row[@"error"]=@"nonfinite_embedding"; return row; }
        [embedding addObject:@(values[i])]; norm2+=values[i]*values[i];
    }
    row[@"embedding"]=embedding;
    row[@"embedding_l2_norm"]=@(sqrt(norm2));
    row[@"faceprint_confidence"]=@([fp confidence]);
    row[@"native_crops"]=[captureRecords copy];
    row[@"faceprint_specifier"]=[[fp originatingRequestSpecifier] description];
    return row;
}

int main(int argc,const char *argv[]) { @autoreleasepool {
    if(argc!=4) { fprintf(stderr,"Usage: faceprint-runner manifest.json output.jsonl crop_directory\n"); return 2; }
    captureRoot=@(argv[3]);installCapture();
    NSError *error=nil;
    NSData *input=[NSData dataWithContentsOfFile:@(argv[1]) options:0 error:&error];
    NSArray *manifest=input?[NSJSONSerialization JSONObjectWithData:input options:0 error:&error]:nil;
    if(![manifest isKindOfClass:[NSArray class]]) { NSLog(@"Invalid manifest: %@",error); return 2; }
    // Refuse overwriting an existing result so an interrupted run stays reviewable.
    FILE *output=fopen(argv[2],"wx");
    if(!output) { perror("output"); return 2; }
    NSUInteger count=0;
    for(NSDictionary *item in manifest) { @autoreleasepool {
        NSDictionary *row;
        @try { row=score(item); }
        @catch(NSException *exception) {
            NSMutableDictionary *failure=[item mutableCopy];
            failure[@"error"]=exception.reason?:@"Objective-C exception"; row=failure;
        }
        NSData *json=[NSJSONSerialization dataWithJSONObject:row options:NSJSONWritingSortedKeys error:&error];
        if(!json || fwrite(json.bytes,1,json.length,output)!=json.length || fputc('\n',output)==EOF) {
            NSLog(@"Output failure: %@",error); fclose(output); return 1;
        }
        fflush(output);
        if(++count%10==0 || row[@"error"]) fprintf(stderr,"%lu/%lu %s%s\n",(unsigned long)count,(unsigned long)manifest.count,
            [item[@"file"] UTF8String],row[@"error"]?[[NSString stringWithFormat:@" ERROR %@",row[@"error"]] UTF8String]:"");
    } }
    fclose(output);
    return 0;
} }
